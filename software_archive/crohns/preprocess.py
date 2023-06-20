import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import io, segmentation, transform
from tqdm import tqdm


def train_val_split(
    images_folder_path: Path,
    gts_folder_path: Path,
    train_ratio: Optional[float] = 0.8,
) -> tuple:
    all_images_paths = np.array(sorted(images_folder_path.glob("*.nii.gz")))
    all_gts_paths = np.array(sorted(gts_folder_path.glob("*.nii.gz")))

    if len(all_images_paths) != len(all_gts_paths):
        raise ValueError("Number of images and ground truths is different")

    shuffled_indices = np.random.permutation(len(all_images_paths))
    train_indices = shuffled_indices[: int(len(all_images_paths) * train_ratio)]
    val_indices = shuffled_indices[int(len(all_images_paths) * train_ratio) :]

    train_images_paths = all_images_paths[train_indices]
    train_gts_paths = all_gts_paths[train_indices]

    val_images_paths = all_images_paths[val_indices]
    val_gts_paths = all_gts_paths[val_indices]

    return train_images_paths, train_gts_paths, val_images_paths, val_gts_paths


def resize(data, size, data_class):
    if data_class == "image":
        order = 3
    elif data_class == "gt":
        order = 0
    else:
        raise ValueError("mode must be either image or gt")

    return transform.resize(
        data,
        (size, size),
        order=order,
        preserve_range=True,
        mode="constant",
        anti_aliasing=True,
    )


def preprocess_image(image_path):
    image = sitk.ReadImage(str(image_path))
    image = sitk.GetArrayFromImage(image)

    lower_bound, upper_bound = np.percentile(image, 0.5), np.percentile(image, 99.5)
    image_processed = np.clip(image, lower_bound, upper_bound)
    image_processed = (
        (image_processed - lower_bound) / (upper_bound - lower_bound) * 255.0
    )
    image_processed[image == 0] = 0
    image_processed = np.uint8(image_processed)

    return image_processed


def preprocess(image_path, gt_path, image_size, model, device):
    image = sitk.ReadImage(str(image_path))
    image = sitk.GetArrayFromImage(image)

    gt = sitk.ReadImage(str(gt_path))
    gt = sitk.GetArrayFromImage(gt)

    # assume that the gt are binary
    gt = np.uint8(gt == 1)

    image_volume, gt_volume, image_embeddings = [], [], []

    if np.sum(gt) > 1000:
        lower_bound, upper_bound = np.percentile(image, 0.5), np.percentile(image, 99.5)
        image_processed = np.clip(image, lower_bound, upper_bound)
        image_processed = (
            (image_processed - lower_bound) / (upper_bound - lower_bound) * 255.0
        )
        image_processed[image == 0] = 0
        image_processed = np.uint8(image_processed)

        z_idx = np.where(gt > 0)[0]
        z_min, z_max = np.min(z_idx), np.max(z_idx)

        for i in range(z_min, z_max):
            gt_slice = resize(gt[i, :, :], image_size, "gt")

            if np.sum(gt_slice) > 100:
                gt_volume.append(gt_slice)
                img_slice = resize(image_processed[i, :, :], image_size, "image")
                # convert to a 3-channel image
                img_slice = np.uint8(np.repeat(img_slice[:, :, None], 3, axis=-1))
                image_volume.append(img_slice)

                if model is not None:
                    sam_tramsform = ResizeLongestSide(model.image_encoder.img_size)
                    resize_img = sam_tramsform.apply_image(img_slice)
                    resize_img_tensor = torch.as_tensor(
                        resize_img.transpose(2, 0, 1)
                    ).to(device)
                    input_image = model.preprocess(resize_img_tensor[None, :, :, :])
                    with torch.no_grad():
                        embedding = model.image_encoder(input_image)
                        image_embeddings.append(embedding.cpu().numpy()[0])

    if model is not None:
        return image_volume, gt_volume, image_embeddings
    else:
        return image_volume, gt_volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRI images")
    parser.add_argument(
        "-i",
        "--nii_path",
        type=str,
        required=True,
        help="path to the MRI images",
    )

    parser.add_argument(
        "-gt",
        "--gt_path",
        type=str,
        required=True,
        help="path to the ground truth",
    )

    parser.add_argument(
        "-o",
        "--npz_path",
        type=str,
        required=True,
        help="path to save the npz files",
    )

    parser.add_argument(
        "--direction",
        type=str,
        choices=["axial", "coronal"],
        default="axial",
        help="direction of the MRI images",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="size of the image",
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help="whether to do inference or not",
    )

    parser.add_argument("--anatomy", type=str, default="Crohns", help="anatomy")

    parser.add_argument("--model_type", type=str, default="vit_b", help="model type")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/SAM/sam_vit_b_01ec64.pth",
        help="checkpoint",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    parser.add_argument("--seed", type=int, default=1706576, help="random seed")

    args = parser.parse_args()

    prefix = "_".join(["MR", args.anatomy, args.direction])

    # split names into training and validation
    base_path = Path(os.getcwd())
    # path for saving the npz files
    npz_path = base_path / args.npz_path
    # load the model
    model = sam_model_registry[args.model_type](args.checkpoint).to(args.device)

    np.random.seed(args.seed)

    if args.inference:
        npz_path_all = npz_path / prefix / "data"
        npz_path_all.mkdir(parents=True, exist_ok=True)

        images_paths = sorted(list((base_path / args.nii_path).glob("*.nii.gz")))
        gts_paths = sorted(list((base_path / args.gt_path).glob("*.nii.gz")))

        for image_path, gt_path in tqdm(
            zip(images_paths, gts_paths), total=len(images_paths)
        ):
            image_volume, gt_volume, _ = preprocess(
                image_path,
                gt_path,
                args.image_size,
                model,
                args.device,
            )

            if len(image_volume) > 1:
                image_volume = np.stack(image_volume, axis=0)
                gt_volume = np.stack(gt_volume, axis=0)
                np.savez_compressed(
                    npz_path_all / (image_path.name[:-7] + ".npz"),
                    imgs=image_volume,
                    gts=gt_volume,
                )
    else:
        (
            train_images_paths,
            train_gts_paths,
            val_images_paths,
            val_gts_paths,
        ) = train_val_split(
            images_folder_path=base_path / args.nii_path,
            gts_folder_path=base_path / args.gt_path,
            train_ratio=0.8,
        )

        npz_path_train = npz_path / prefix / "train"
        npz_path_val = npz_path / prefix / "val"

        npz_path_train.mkdir(parents=True, exist_ok=True)
        npz_path_val.mkdir(parents=True, exist_ok=True)

        # preprocess the training images
        for image_path, gt_path in tqdm(
            zip(train_images_paths, train_gts_paths), total=len(train_images_paths)
        ):
            image_volume, gt_volume, image_embeddings = preprocess(
                image_path,
                gt_path,
                args.image_size,
                model,
                args.device,
            )

            if len(image_volume) > 1:
                image_volume = np.stack(image_volume, axis=0)
                gt_volume = np.stack(gt_volume, axis=0)
                image_embeddings = np.stack(image_embeddings, axis=0)
                np.savez_compressed(
                    npz_path_train / (image_path.name[:-7] + ".npz"),
                    imgs=image_volume,
                    gts=gt_volume,
                    img_embeddings=image_embeddings,
                )

                # save an image for sanity check
                idx = np.random.randint(0, len(image_volume))
                image_slice = image_volume[idx, :, :, :]
                gt_slice = gt_volume[idx, :, :]
                boundary = segmentation.find_boundaries(gt_slice, mode="inner")
                image_slice[boundary, :] = [255, 0, 0]
                io.imsave(
                    str(npz_path_train) + ".png",
                    image_slice,
                    check_contrast=False,
                )

        # preprocess the validation images
        for image_path, gt_path in tqdm(
            zip(val_images_paths, val_gts_paths), total=len(val_images_paths)
        ):
            image_volume, gt_volume, image_embeddings = preprocess(
                image_path,
                gt_path,
                args.image_size,
                model,
                args.device,
            )

            if len(image_volume) > 1:
                image_volume = np.stack(image_volume, axis=0)
                gt_volume = np.stack(gt_volume, axis=0)
                image_embeddings = np.stack(image_embeddings, axis=0)
                np.savez_compressed(
                    npz_path_val / (image_path.name[:-7] + ".npz"),
                    imgs=image_volume,
                    gts=gt_volume,
                    img_embeddings=image_embeddings,
                )

                # save an image for sanity check
                idx = np.random.randint(0, len(image_volume))
                image_slice = image_volume[idx]
                gt_slice = gt_volume[idx]
                boundary = segmentation.find_boundaries(gt_slice, mode="inner")
                image_slice[boundary, :] = [255, 0, 0]
                io.imsave(
                    str(npz_path_val) + ".png",
                    image_slice,
                    check_contrast=False,
                )
