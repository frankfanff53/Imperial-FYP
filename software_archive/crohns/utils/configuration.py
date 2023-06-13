import os
import argparse
from pathlib import Path


from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--stored-path",
        type=str,
        required=True,
        help="Path to the stored directory containing the segmentation dataset.",
    )

    args = parser.parse_args()
    base_path = Path(os.getcwd())

    # create nnuNet_raw, nnuNet_preprocessed, nnuNet_results folder under stored_path
    data_dir = base_path / args.stored_path
    nnuNet_raw_path = data_dir / "nnUNet_raw"
    nnuNet_preprocessed_path = data_dir / "nnUNet_preprocessed"
    nnuNet_results_path = data_dir / "nnUNet_results"

    maybe_mkdir_p(nnuNet_raw_path)
    maybe_mkdir_p(nnuNet_preprocessed_path)
    maybe_mkdir_p(nnuNet_results_path)

    # set up environment variables
    os.environ["nnUNet_raw"] = str(nnuNet_raw_path)
    os.environ["nnUNet_preprocessed"] = str(nnuNet_preprocessed_path)
    os.environ["nnUNet_results"] = str(nnuNet_results_path)

    # validate environment variables
    print("nnUNet_raw: ", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed: ", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results: ", os.environ["nnUNet_results"])
