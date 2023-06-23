import argparse
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from crohns import compute_dice_coefficient, get_bbox_from_mask, show_mask


def finetune_model_predict(
    img_np, box_np, sam_trans, sam_model_tune, device="cuda:0"
):
    """ Predict the segmentation mask for the given image and bounding box
    using the finetuned model.

    Args:
        img_np (np.ndarray): The image as a numpy array.
        box_np (np.ndarray): The bounding box as a numpy array.
        sam_trans: The transform function for transforming inputs
            and subsquently feeding them to the SAM model.
        sam_model_tune: The finetuned model.
        device (str): The device to use for prediction.

    Returns:
        np.ndarray: The predicted segmentation mask.
    """

    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
        device
    )
    input_image = sam_model_tune.preprocess(resize_img_tensor[None, :, :, :])
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device))
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        # Ablation study for testing the effect of localisation
        # using bounding box
        if "ablation" in args.task_name:
            box_torch = None

        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert to binary mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--task-name", type=str, required=True, help="task name"
    )

    parser.add_argument(
        "--model-type", type=str, default="vit_b", help="model type"
    )

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

    ori_sam_model = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint
    ).to(args.device)
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
            medsam_seg = finetune_model_predict(
                img, bbox, sam_trans, medsam_model, args.device
            )
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

    print("Average DSC: {:.4f}".format(np.mean(dscs)))
    print("Median DSC: {:.4f}".format(np.median(dscs)))
    print("Std DSC: {:.4f}".format(np.std(dscs)))
    print("Max DSC: {:.4f}".format(np.max(dscs)))
