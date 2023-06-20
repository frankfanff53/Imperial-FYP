import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def finetune_model_predict(img_np, box_np, sam_trans, sam_model_tune, device="cuda:0"):
    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(
        resize_img_tensor[None, :, :, :]
    )  # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(
            input_image.to(device)
        )  # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg


def get_bbox_from_mask(mask):
    """Returns a bounding box from a mask"""
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--npz-path",
        type=str,
        required=True,
        help="path to the npz file",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="work_dir",
        help="path to the model directory",
    )

    parser.add_argument("--task-name", type=str, required=True, help="task name")

    parser.add_argument("--model-type", type=str, default="vit_b", help="model type")

    parser.add_argument(
        "--random-seed",
        type=int,
        default=1706576,
        help="random seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    base_path = Path(os.getcwd())
    npz_path = base_path / args.npz_path
    mask_save_path = (base_path / args.npz_path).parent / "masks"
    mask_save_path.mkdir(exist_ok=True)

    # load the model
    sam_model_tune = sam_model_registry[args.model_type](
        checkpoint=str(Path(args.model_dir) / args.task_name / "sam_model_best.pth")
    ).to(args.device)

    sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

    for i, npz_file in enumerate(sorted(npz_path.glob("*.npz"))):
        npz_data = np.load(npz_file)
        ori_imgs = npz_data["imgs"]
        ori_gts = npz_data["gts"]
        print(f"Shape of the images: {ori_imgs.shape}")

        sam_segs = []
        sam_bboxes = []

        for j, ori_img in enumerate(ori_imgs):
            gt2D = ori_gts[j]
            # get bounding box from gt
            bbox = get_bbox_from_mask(gt2D)
            # predict segmentation mask
            sam_seg = finetune_model_predict(
                ori_img,
                bbox,
                sam_trans,
                sam_model_tune,
                device=args.device,
            )
            sam_segs.append(sam_seg)
            sam_bboxes.append(bbox)

        print(f"Shape of the segmentation masks: {np.array(sam_segs).shape}")
        # save the segmentation mask
        labels_name = "_".join(npz_file.name.split("_")[:-1]) + ".npz"
        np.savez_compressed(
            mask_save_path / labels_name,
            gts=sam_segs,
        )

        img_id = np.random.randint(0, len(ori_imgs))
        # make one plot with size (5, 5)
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.imshow(ori_imgs[img_id])
        show_mask(sam_segs[img_id], axes)
        show_box(sam_bboxes[img_id], axes)
        axes.set_title("Generated weakly-supervised segmentation mask")
        axes.axis("off")
        fig.savefig(
            mask_save_path.parent / "weak_mask.png", bbox_inches="tight", dpi=300
        )
        plt.close(fig)
