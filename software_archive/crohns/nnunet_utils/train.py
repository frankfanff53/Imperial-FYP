import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # script format: python train.py --dim <dimension> --id <dataset-id> --fold <start-fold> <end-fold> --device <device-id> --trainer <trainer-name>
    parser.add_argument(
        "--id",
        required=True,
        help="Dataset ID or Name.",
    )

    parser.add_argument(
        "--dim",
        type=str,
        required=True,
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade"],
        help="Dimension of the dataset.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        required=True,
        help="Fold ID.",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "mps"],
        help="Device for training. CUDA is recommended.",
    )

    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    dataset_id = args.id
    folds = args.fold
    dimension = args.dim
    device = args.device
    trainer = args.trainer

    for fold in folds:
        commands = [
            "nnUNetv2_train",
            dataset_id,
            dimension,
            str(fold),
            "-device",
            device,
            "-tr",
            trainer,
            "--npz",
        ]
        print(" ".join(commands))
        result = subprocess.run(commands)
        if result.returncode != 0:
            print(f"Training failed for {dataset_id} fold {fold}.")
            break
