import argparse
import logging
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from crohns import (
    compute_dice_coefficient,
    get_bbox_from_centerline,
    get_centerline_points,
)
from crohns.baseline import (
    apply_denoising,
    apply_laplacian_sharpening,
    apply_morphological_hole_closing,
    apply_n4_correction,
    apply_slic,
    apply_voting_binary_hole_filling,
    crop_image,
    get_slic_segments_on_centerline,
)


def get_mask_on_full_image(
    weak_mask,
    image_shape,
    index,
    size,
) -> sitk.Image:
    xmin, ymin, zmin = index
    width, height, depth = size
    xmax, ymax, zmax = xmin + width, ymin + height, zmin + depth
    full_seg_arr = np.zeros(image_shape, dtype=np.float32)
    full_seg_arr[
        xmin:xmax,
        ymin:ymax,
        zmin:zmax,
    ] = sitk.GetArrayFromImage(weak_mask)
    return sitk.GetImageFromArray(full_seg_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image-folder-path",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "-gt",
        "--ground-truth-folder-path",
        type=str,
        required=True,
        help="Path to the ground truth segmentation folder.",
    )

    parser.add_argument(
        "-c",
        "--centerline-folder-path",
        type=str,
        required=True,
        help="Path to the centerline folder.",
    )

    parser.add_argument(
        "-o",
        "--output-folder-path",
        type=str,
        help="Path to store the output segmentation.",
    )

    args = parser.parse_args()

    base_path = Path(os.getcwd())

    image_folder = base_path / args.image_folder_path
    label_folder = base_path / args.ground_truth_folder_path
    centerline_folder = base_path / args.centerline_folder_path
    output_path = base_path / args.output_folder_path
    # create output directory if not exists
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_folder.glob("*.nii.gz"))
    label_paths = sorted(label_folder.glob("*.nii.gz"))
    centerline_paths = sorted(centerline_folder.glob("*.xml"))

    # logger settings
    logging.basicConfig(
        level=logging.INFO,
        filename=str(output_path / "weak_label_generation.txt"),
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger("slic_weak_label_generation")

    for image_path, label_path, centerline_path in zip(
        image_paths, label_paths, centerline_paths
    ):
        fname = image_path.name[: -len(".nii.gz")]
        # weak label generation pipeline
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        img_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        # get centerline points and determine bounding box
        centerline_points = get_centerline_points(centerline_path)
        index, size = get_bbox_from_centerline(centerline_points)

        # crop the image to extract the region of interest
        try:
            cropped_image = crop_image(image, index, size)
        except RuntimeError:
            raise RuntimeError(f"Failed to crop image {fname}")

        # apply preprocessing
        corrected_image = apply_n4_correction(
            cropped_image,
            max_iter=10,
            control_points=4,
        )
        denoised_image = apply_denoising(
            cropped_image,
            time_step=5e-3,
            iterations=100,
        )
        sharpened_image = apply_laplacian_sharpening(cropped_image)

        # apply SLIC segmentation
        slic_mask = apply_slic(
            sharpened_image,
            max_iter=50,
            super_grid_size=6,
        )

        # extract weak label on the centerline
        points_in_ROI = centerline_points - np.array(index)
        slic_mask = get_slic_segments_on_centerline(
            slic_mask,
            points_in_ROI,
        )

        # apply post-processing
        binary_mask = apply_voting_binary_hole_filling(
            slic_mask,
            radius=2,
            max_iter=50,
        )
        morph_mask = apply_morphological_hole_closing(
            binary_mask,
            kernel_radius=7,
            kernel_type=sitk.sitkBall,
        )

        # fill the weak label to the full image
        full_mask = get_mask_on_full_image(
            morph_mask,
            img_arr.shape,
            index,
            size,
        )

        full_mask_arr = sitk.GetArrayFromImage(full_mask)

        # save the weak label
        sitk.WriteImage(full_mask, output_path / f"{fname}.nii.gz")

        dsc = compute_dice_coefficient(label_arr > 0, full_mask_arr > 0)
        logger.info(f"Dice coefficient for {fname}: {dsc:.4f}")
