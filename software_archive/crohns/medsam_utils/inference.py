import argparse
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from crohns import get_bbox_from_mask, show_box, show_mask
from crohns.medsam_utils import finetune_model_predict

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

    parser.add_argument(
        "--task-name", type=str, required=True, help="task name"
    )

    parser.add_argument(
        "--model-type", type=str, default="vit_b", help="model type"
    )

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
        checkpoint=str(
            Path(args.model_dir) / args.task_name / "sam_model_best.pth"
        )
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
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.imshow(ori_imgs[img_id])
        show_mask(sam_segs[img_id], axes)
        show_box(sam_bboxes[img_id], axes)
        axes.set_title("Generated weakly-supervised segmentation mask")
        axes.axis("off")
        fig.savefig(
            mask_save_path.parent / "weak_mask.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
