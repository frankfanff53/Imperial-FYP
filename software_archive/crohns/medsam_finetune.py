import argparse
import os
from pathlib import Path

import monai
import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(os.path.join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d["gts"] for d in self.npz_data])
        self.img_embeddings = np.vstack([d["img_embeddings"] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
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
        "--model-dir", type=str, default="work_dir", help="path to the model directory"
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
        help="patience",
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
        # weight_decay=1e-5,
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
                box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                box_torch = torch.tensor(box).float().to(args.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                # embeddings
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
                box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
                box_torch = torch.tensor(box).float().to(args.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]
                # embeddings
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
        print(f"Epoch {epoch} train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

        torch.save(model.state_dict(), model_save_path / "sam_model_lateset.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path / "sam_model_best.pth")
        # early stopping
        if val_loss < early_stopping_val_loss:
            early_stopping_val_loss = val_loss
            early_stopping_start_epoch = epoch
        if epoch - early_stopping_start_epoch > early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_losses_all, val_losses_all = zip(*losses)
    plt.plot(train_losses_all, label="Train Loss")
    plt.plot(val_losses_all, label="Validation Loss")
    plt.title("Dice + Cross Entropy Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(model_save_path / "loss.png", dpi=300)
    plt.show()
