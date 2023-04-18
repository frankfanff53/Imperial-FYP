# Tackling Crohn's Disease

## How to set up the environment

### Set up Locally (with Mini-Conda)

```bash
conda create -n myenv python=3.10
conda activate myenv
```

#### Install Dependencies

```bash
make install
```

#### Install nnU-Net

You can install it via `pip`:

```bash
pip install nnunetv2
```

Or install it as an integrative framework by cloning the original repo:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

#### Set Environmental Variables

nnU-Net requires three environmental variables to be set:

- `nnUNet_raw`: Specify the location of the raw data
- `nnUNet_preprocessed`: Specify the location of the preprocessed data
- `nnUNet_results`: Specify the location of the trained models

We use the following command to create folders to store the data, and temporarily set the three environment variables above:

```python
cd software_archive
python crohns/utils/configuration.py -p data
```

Here the `-p` flag specifies the folder you want to store the segmentation dataset, and the `nnUNet_*` folders will be created under this folder.


#### Convert the Training dataset

We use the following command to convert the training dataset to the format that accepted by nnUNet:

```bash
# Usage: python crohns/utils/convert_dataset.py -i ORIGINAL-DATASET-PATH --dataset-type TRAIN/VAL/TEST --taks-name YOUR-TASK-NAME --task-id YOUR-UNIQUE-ID
python crohns/utils/convert_dataset.py -i data/BraTS2020_Training --dataset-type train --task-name BraTS2020 --task-id 1
```

The arguments for the scripts have the following meanings:

- The `-i` flag specifies the path of the original dataset.
- The `--dataset-type` flag specifies the type of the provided dataset, it should be one of the **train, val** or **test**.
- The `--task-name` is user-customised, you can just put one unique identifier for your segmentation task.
- The `--task-id` is also user-customised and it is bonded with the value of `--task-name` to identify your converted dataset. Make sure the task id is a unique identifier for your dataset.

#### Verify the settings

You can run the following command to verify your converted dataset (I will continue to use the example settings above):

```bash
# USAGE: nnUNetv2_plan_and_preprocess -d TASK-ID --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Successful settings will have the following outputs:

```txt
Fingerprint extraction...
Dataset001_BraTS2020
Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> as reader/writer

####################
verify_dataset_integrity Done. 
If you didn't see any error messages then your dataset is most likely OK!
####################

Experiment planning...
2D U-Net configuration:
...

Using <class 'nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO'> as reader/writer
3D fullres U-Net configuration:
...

Plans were saved to /YOUR-PATH/nnUNet_preprocessed/Dataset001_BraTS2020/nnUNetPlans.json
Preprocessing...
Preprocessing dataset Dataset001_BraTS2020
Configuration: 2d...
...
Configuration: 3d_fullres...
...
Configuration: 3d_lowres...
INFO: Configuration 3d_lowres not found in plans file nnUNetPlans.json of dataset Dataset001_BraTS2020. Skipping.
```

The `3d_lowres` will be sometimes omitted due to the small image size in the provided dataset. Once you get the output above, you are free for training.

### Training

TBC
