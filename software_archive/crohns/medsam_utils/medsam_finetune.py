import argparse
import os
from pathlib import Path

import monai
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from crohns import get_bbox_from_mask


class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [
            np.load(os.path.join(data_root, f)) for f in self.npz_files
        ]
        self.ori_gts = np.vstack([d["gts"] for d in self.npz_data])
        self.img_embeddings = np.vstack(
            [d["img_embeddings"] for d in self.npz_data]
        )

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        bboxes = get_bbox_from_mask(gt2D)
        # convert img embedding, mask, bounding box to torch tensor
        return (
            torch.tensor(img_embed).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
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
        "--batch-size",
        type=int,
        default=64,
        help="batch size",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-5,
        help="learning rate",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="number of epochs",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="patience for early stopping",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    base_path = Path(os.getcwd())

    model_save_path = base_path / args.model_dir / args.task_name
    model_save_path.mkdir(parents=True, exist_ok=True)
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
        args.device
    )

    optimizer = torch.optim.Adam(
        model.mask_decoder.parameters(),
        lr=args.learning_rate,
    )

    loss_func = monai.losses.DiceLoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
    )

    train_dataset = NpzDataset(args.train_data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_dataset = NpzDataset(args.val_data_root)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    losses = []
    best_loss = 1e10

    early_stopping_patience = args.patience
    early_stopping_start_epoch = -1
    early_stopping_val_loss = 1e10
    for epoch in range(args.num_epochs):
        train_losses = []

        model.train()
        for image_embedding, gt2D, boxes in tqdm(train_loader):
            with torch.no_grad():
                box_np = boxes.numpy()
                sam_trans = ResizeLongestSide(model.image_encoder.img_size)
                box = sam_trans.apply_boxes(
                    box_np, (gt2D.shape[-2], gt2D.shape[-1])
                )
                box_torch = torch.tensor(box).float().to(args.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                # use bounding box to get sparse and dense embeddings
                # which helps in enhacing the prediction
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embedding.to(args.device),
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            loss = loss_func(mask_predictions, gt2D.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # evaluate on validation set
        val_losses = []
        model.eval()
        for image_embedding, gt2D, boxes in tqdm(val_loader):
            with torch.no_grad():
                box_np = boxes.numpy()
                sam_trans = ResizeLongestSide(model.image_encoder.img_size)
                box = sam_trans.apply_boxes(
                    box_np, (gt2D.shape[-2], gt2D.shape[-1])
                )
                box_torch = torch.tensor(box).float().to(args.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]

                if "ablation" in args.task_name:
                    box_torch = None
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embedding.to(args.device),
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            loss = loss_func(mask_predictions, gt2D.to(args.device))
            val_losses.append(loss.item())
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        losses.append((train_loss, val_loss))
        print(
            f"Epoch {epoch} train loss: {train_loss:.4f}, val loss: {val_loss:.4f}"
        )

        torch.save(
            model.state_dict(), model_save_path / "sam_model_lateset.pth"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(), model_save_path / "sam_model_best.pth"
            )
        # early stopping
        if val_loss < early_stopping_val_loss:
            early_stopping_val_loss = val_loss
            early_stopping_start_epoch = epoch
        if epoch - early_stopping_start_epoch > early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
