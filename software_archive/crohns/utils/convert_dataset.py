import argparse
import os
import shutil
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subdirs
from nnunetv2.dataset_conversion.Dataset137_BraTS21 import (
    copy_BraTS_segmentation_and_convert_labels_to_nnUNet,
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm

MODALITY_ENCODINGS = {
    'T1': 0,
    'T1ce': 1,
    'T2': 2,
    'Flair': 3,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the segmentation dataset.",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        default="train",
        choices=["train", "val", "test"],
        help="Type of the dataset.",
    )

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID.",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name.",
    )

    args = parser.parse_args()
    input_dataset_dir = Path(os.getcwd()) / args.input_dir

    # Define the output directory to store the converted dataset.
    converted_dataset_dir_name = f"Dataset{args.task_id:03d}_{args.task_name}"
    converted_dataset_dir = Path(nnUNet_raw) / converted_dataset_dir_name

    # Specify the directory to store the images and labels.
    dir_suffix = "Tr" if args.dataset_type == "train" else "Ts"
    images_dir = converted_dataset_dir / f"images{dir_suffix}"
    labels_dir = converted_dataset_dir / f"labels{dir_suffix}"

    # Create the directories.
    maybe_mkdir_p(images_dir)
    maybe_mkdir_p(labels_dir)

    # Get the directories containing the cases of the dataset.
    case_dirs = subdirs(input_dataset_dir, join=False)

    # Copy the images and labels to the output directory.
    for case_dir_id in tqdm(case_dirs, desc="Converting dataset"):
        input_case_dir = input_dataset_dir / case_dir_id

        # copy different modalities to imagesTr
        for modality in MODALITY_ENCODINGS.keys():
            modality_file_name = f"{case_dir_id}_{modality.lower()}.nii.gz"
            converted_modality_file_name = f"{case_dir_id}" \
                f"_{MODALITY_ENCODINGS[modality]:04}.nii.gz"
            modality_file = input_case_dir / modality_file_name
            if modality_file.exists():
                shutil.copy(
                    modality_file,
                    images_dir / converted_modality_file_name
                )

        # copy the segmentation to labelsTr
        segmentation_file_name = f"{case_dir_id}_seg.nii.gz"
        converted_segmentation_file_name = f"{case_dir_id}.nii.gz"
        segmentation_file = input_case_dir / segmentation_file_name
        if segmentation_file.exists():
            copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
                in_file=segmentation_file,
                out_file=labels_dir / converted_segmentation_file_name
            )

    # Create the dataset json file.
    generate_dataset_json(
        output_folder=converted_dataset_dir,
        channel_names={encoding: modality for modality, encoding in MODALITY_ENCODINGS.items()},
        labels={
            "background": 0,
            "whole tumor": (1, 2, 3),
            "tumor core": (2, 3),
            "enhancing tumor": (3,),
        },
        num_training_cases=len(case_dirs),
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
        license="see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
        reference="see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
        dataset_release_date="1.0",
    )
