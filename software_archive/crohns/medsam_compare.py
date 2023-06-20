import argparse
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-root",
        type=str,
        required=True,
        help="path to the training data root",
    )

    parser.add_argument(
        "--val-data-root",
        type=str,
        required=True,
        help="path to the validation data root",
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
        "--checkpoint",
        type=str,
        default="work_dir/SAM/sam_vit_b_01ec64.pth",
        help="path to the checkpoint",
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

    model_save_path = base_path / args.model_dir / args.task_name

    medsam_model = sam_model_registry[args.model_type](
        checkpoint=str(model_save_path / "sam_model_best.pth")
    ).to(args.device)

    ori_sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
        args.device
    )
    ori_sam_predictor = SamPredictor(ori_sam_model)
    sam_trans = ResizeLongestSide(ori_sam_model.image_encoder.img_size)

    npz_ts_path = base_path / args.val_data_root
    test_npzs = sorted(npz_ts_path.glob("*.npz"))

    # random select a test case
    dscs = []
    for test_npz in test_npzs:
        npz = np.load(os.path.join(npz_ts_path, test_npz))
        imgs = npz["imgs"]
        gts = npz["gts"]

        ori_sam_segs = []
        medsam_segs = []
        bboxes = []
        for img, gt in zip(imgs, gts):
            bbox = get_bbox_from_mask(gt)
            bboxes.append(bbox)
            # predict the segmentation mask using the original SAM model
            ori_sam_predictor.set_image(img)
            ori_sam_seg, _, _ = ori_sam_predictor.predict(
                point_coords=None, box=bbox, multimask_output=False
            )
            ori_sam_segs.append(ori_sam_seg[0])

            # predict the segmentation mask using the fine-tuned model
            H, W = img.shape[:2]
            resize_img = sam_trans.apply_image(img)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
                args.device
            )
            input_image = medsam_model.preprocess(resize_img_tensor[None, :, :, :])
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(input_image.to(args.device))
                # convert box to 1024x1024 grid
                bbox = sam_trans.apply_boxes(bbox, (H, W))
                box_torch = torch.as_tensor(bbox, dtype=torch.float, device=args.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                
                if "ablation" in args.task_name:
                    box_torch = None
                sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
                medsam_seg_prob, _ = medsam_model.mask_decoder(
                    image_embeddings=image_embedding.to(args.device),
                    image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
                # convert soft mask to hard mask
                medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                medsam_segs.append(medsam_seg)

        ori_sam_segs = np.stack(ori_sam_segs, axis=0)
        medsam_segs = np.stack(medsam_segs, axis=0)
        ori_sam_dsc = compute_dice_coefficient(gts > 0, ori_sam_segs > 0)
        medsam_dsc = compute_dice_coefficient(gts > 0, medsam_segs > 0)
        if medsam_dsc < 0.5:
            continue
        print(
            "Original SAM DSC: {:.4f}".format(ori_sam_dsc),
            "MedSAM DSC: {:.4f}".format(medsam_dsc),
        )
        dscs.append(medsam_dsc)

        img_id = int(imgs.shape[0] / 2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(imgs[img_id])
        show_mask(gts[img_id], ax)
        ax.axis("off")
        fig.savefig(
            model_save_path / "{}_gt.png".format(test_npz.name[:-9]),
            dpi=300,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(imgs[img_id])
        show_mask(ori_sam_segs[img_id], ax)
        ax.axis("off")
        fig.savefig(
            model_save_path / "{}_ori_sam.png".format(test_npz.name[:-9]),
            dpi=300,
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(imgs[img_id])
        show_mask(medsam_segs[img_id], ax)
        ax.axis("off")
        fig.savefig(
            model_save_path / "{}_medsam.png".format(test_npz.name[:-9]),
            dpi=300,
        )

    # write to log
    with open(model_save_path / "dsc_log.txt", "w") as f:
        f.write("Average DSC: {:.4f}\n".format(np.mean(dscs)))
        f.write("Median DSC: {:.4f}\n".format(np.median(dscs)))
        f.write("Std DSC: {:.4f}\n".format(np.std(dscs)))
        f.write("Max DSC: {:.4f}\n".format(np.max(dscs)))

    print("Average DSC: {:.4f}".format(np.mean(dscs)))
    print("Median DSC: {:.4f}".format(np.median(dscs)))
    print("Std DSC: {:.4f}".format(np.std(dscs)))
    print("Max DSC: {:.4f}".format(np.max(dscs)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.boxplot(dscs)
    ax.set_xticklabels(["Dice Score"])
    ax.set_title("Dice Score Distribution for Generated Weak Masks", fontsize=16, fontweight="bold")
    fig.savefig(
        model_save_path / "dsc_dist.png", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.close(fig)
