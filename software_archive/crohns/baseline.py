import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


def get_centerline_points(centerline_path):
    crohns_centerline = ET.parse(centerline_path)
    root = crohns_centerline.getroot()
    centerline_points = []
    for path in root:
        if "name" not in path.attrib:
            continue

        for point in path:
            centerline_points.append(
                [int(point.attrib["x"]), int(point.attrib["y"]), int(point.attrib["z"])]
            )

    crohns_centerline_size = int(len(centerline_points) * 0.2)
    return np.array(centerline_points[:crohns_centerline_size])


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def get_bbox_from_centerline(centerline_points):
    min_x = np.min(centerline_points[:, 0])
    max_x = np.max(centerline_points[:, 0])
    min_y = np.min(centerline_points[:, 1])
    max_y = np.max(centerline_points[:, 1])
    min_z = np.min(centerline_points[:, 2])
    max_z = np.max(centerline_points[:, 2])

    # the top left corner of the bounding box
    index = (int(min_x), int(min_y), int(min_z))
    # the size of the bounding box
    size = (int(max_x - min_x + 1), int(max_y - min_y + 1), int(max_z - min_z + 1))
    return index, size


def show_mask(mask, ax):
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def crop_image(img, index, size):
    return sitk.RegionOfInterest(img, size, index)


def process_cropped_image(cropped_image):
    myFilterCastImage = sitk.CastImageFilter()

    # Set output pixel type (float32)
    myFilterCastImage.SetOutputPixelType(sitk.sitkFloat32)
    cropped_img_float32 = myFilterCastImage.Execute(cropped_image)

    # Apply N4 bias field correction to the image
    n4_correction = sitk.N4BiasFieldCorrectionImageFilter()
    n4_correction.SetConvergenceThreshold(1e-3)
    n4_correction.SetMaximumNumberOfIterations((10, 10, 10, 10))
    n4_correction.SetBiasFieldFullWidthAtHalfMaximum(0.1)
    n4_correction.SetWienerFilterNoise(0.01)
    n4_correction.SetNumberOfHistogramBins(100)
    n4_correction.SetNumberOfControlPoints((4, 4, 4))
    n4_correction.SetSplineOrder(3)
    corrected_img = n4_correction.Execute(cropped_img_float32)

    # Denoising using curvature driven flow
    cflow = sitk.CurvatureFlowImageFilter()
    cflow.SetTimeStep(0.05)
    cflow.SetNumberOfIterations(100)
    denoised_img = cflow.Execute(corrected_img)

    # Laplacian sharpening
    lp_sharp = sitk.LaplacianSharpeningImageFilter()
    sharpened_edges_image = lp_sharp.Execute(denoised_img)

    slic = sitk.SLICImageFilter()
    slic.SetMaximumNumberOfIterations(50)
    slic.SetSuperGridSize((6, 6, 6))
    slic.SetSpatialProximityWeight(5.0)
    slic.SetEnforceConnectivity(True)
    slic.SetInitializationPerturbation(True)
    return slic.Execute(sharpened_edges_image)


def get_slic_segments_using_coordinates(seg: sitk.Image, points: list):
    """
    Using a SLIC-segmented image and centreline co-ordinates, select the superpixel clusters which
    lie on the centreline. 

    :param seg: SLIC-segmented SimpleITK image
    :param points: list of centreline co-ordinates
    :return: selected superpixel clusters, as a SimpleITK image
    """
    def apply_intensity_mask(arr, intensity_mask):
        arr[intensity_mask] = 0

    def generate_intensity_mask(arr, required_intensities):
        return ~np.isin(arr, required_intensities)

    required_intensities = []
    arr = sitk.GetArrayFromImage(seg)

    for point in points:
        x, y, z = point
        # numpy requires (z, y, x) form
        intensity = arr[int(z), int(y), int(x)]
        if intensity not in required_intensities:
            required_intensities.append(intensity)

    intensity_mask = generate_intensity_mask(arr, required_intensities)
    apply_intensity_mask(arr, intensity_mask)

    return sitk.GetImageFromArray(arr)


def get_weak_mask(cropped_seg, centerline_points, index):
    points_in_cropped_image = centerline_points - np.array(index)
    out = get_slic_segments_using_coordinates(cropped_seg, points_in_cropped_image)

    # Do voting binary hole filling
    voting_ibhole_filling = sitk.VotingBinaryHoleFillingImageFilter()
    voting_ibhole_filling.SetBackgroundValue(0.0)
    voting_ibhole_filling.SetForegroundValue(1.0)
    voting_ibhole_filling = sitk.VotingBinaryIterativeHoleFillingImageFilter()
    voting_ibhole_filling.SetRadius((2, 2, 2))
    voting_ibhole_filling.SetMaximumNumberOfIterations(50)
    voting_ibhole_filling.SetMajorityThreshold(1)
    seg_after_binary_hole_filling = voting_ibhole_filling.Execute(out)

    # Do morphological hole closing
    morph_closing = sitk.BinaryMorphologicalClosingImageFilter()
    morph_closing.SetSafeBorder(True)
    morph_closing.SetKernelRadius((7, 7, 7))
    morph_closing.SetKernelType(sitk.sitkBall)
    return morph_closing.Execute(seg_after_binary_hole_filling)


def get_mask_on_full_image(img_arr, weak_mask, index, size):
    xmin, ymin, zmin = index
    width, height, depth = size
    full_seg_arr = np.zeros_like(img_arr)
    full_seg_arr[zmin:zmin + depth, ymin:ymin + height, xmin:xmin + width] = sitk.GetArrayFromImage(weak_mask)
    return sitk.GetImageFromArray(full_seg_arr)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    # centerline_folder_name = "Crohns2023AxialCenterlines"
    # image_folder_name = "Crohns2023Axial"
    # label_folder_name = "Crohns2023AxialLabels"
    centerline_folder_name = "Crohns2023CoronalCenterlines"
    image_folder_name = "Crohns2023Coronal"
    label_folder_name = "Crohns2023CoronalLabels"
    crohns_data_folder = base_path / "processed_crohns"
    centerline_folder = crohns_data_folder / centerline_folder_name
    image_folder = crohns_data_folder / image_folder_name
    label_folder = crohns_data_folder / label_folder_name
    centerline_fnames = set([f.name[:-4] for f in centerline_folder.glob("*.xml")])
    image_fnames = set([f.name[:-7] for f in image_folder.glob("*.nii.gz")])
    label_fnames = set([f.name[:-7] for f in label_folder.glob("*.nii.gz")])

    # Get the intersection of the three sets
    fnames = centerline_fnames.intersection(image_fnames).intersection(label_fnames)
    dscs = []
    for fname in fnames:
        image_path = image_folder / (fname + ".nii.gz")
        label_path = label_folder / (fname + ".nii.gz")
        centerline_path = centerline_folder / (fname + ".xml")

        # Load the image, label and centerline
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        img_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label)

        centerline_points = get_centerline_points(centerline_path)
        index, size = get_bbox_from_centerline(centerline_points)
        try:
            cropped_image = crop_image(image, index, size)
        except RuntimeError:
            print(f"{fname} failed to crop")
            continue
        cropped_seg = process_cropped_image(cropped_image)
        weak_mask = get_weak_mask(cropped_seg, centerline_points, index)
        full_mask = get_mask_on_full_image(img_arr, weak_mask, index, size)
        full_mask_arr = sitk.GetArrayFromImage(full_mask)

        dsc = compute_dice_coefficient(label_arr > 0, full_mask_arr > 0)
        print(f"{fname} Dice coefficient: {dsc:.4f}")
        dscs.append(dsc)

        # fig, ax = plt.subplots(figsize=(5, 5))
        # idx = index[2] + 1
        # ax.imshow(img_arr[idx], cmap="gray")
        # show_mask(label_arr[idx], ax)
        # ax.axis("off")
        # fig.savefig(f"{fname}_gt.png")
        # plt.close(fig)

        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.imshow(img_arr[idx], cmap="gray")
        # show_mask(full_mask_arr[idx], ax)
        # ax.axis("off")
        # fig.savefig(f"{fname}_pred_dsc_{dsc:.4f}.png")
        # plt.close(fig)

    with open(f"{image_folder_name}.txt", "w") as f:
        f.write(f"Average Dice coefficient: {np.mean(dscs):.4f}\n")
        f.write(f"Standard deviation: {np.std(dscs):.4f}\n")
        f.write(f"Median: {np.median(dscs):.4f}\n")
        f.write(f"Minimum: {np.min(dscs):.4f}\n")

    print(f"Average Dice coefficient: {np.mean(dscs):.4f}")
    print(f"Standard deviation: {np.std(dscs):.4f}")
    print(f"Median: {np.median(dscs):.4f}")
    print(f"Minimum: {np.min(dscs):.4f}")
    print(f"Maximum: {np.max(dscs):.4f}")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(dscs)
    ax.set_ylabel("Dice coefficient")
    ax.set_xticklabels([""])
    fig.savefig(f"{image_folder_name}_boxplot.png")
    plt.close(fig)
