# Tackling Crohn's Disease 2023

## How to set up the environment

### Set up Locally (with Mini-Conda)

```bash
conda create -n myenv python=3.10
conda activate myenv
```

#### Install Dependencies

I highly recommend to start this project on a CUDA-enabled machine (with at least an RTX 2070/80 for training), since CUDA-enabled machine is more supported for the nnU-Net architecture.

```bash
# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=<cuda-toolkit-version> -c pytorch -c nvidia

# Install visualisation tool
conda install matplotlib

# Install SegmentAnything package
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install the library itself
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

We need to add the three variables above to your environment variables, by modifying your shell configuration file (e.g. `.bashrc` or `.zshrc`), or go to your OS settings (especially for Windows).

#### Convert the Training dataset
For nnU-Net you need to use a dataset with strict formats

We use the following command to convert the training dataset to the format that accepted by nnUNet. For Crohn's data, you need to ensure there is a folder representing each case's (subject, patient, whatever) image data and (weak) label, as it shows below:
```bash
Crohns2023
|- Crohns2023_001 # This is the folder for patient 001, containing data and labels
|  |- Crohns2023_001_T2.nii.gz # This represents the image data with modality T2
|  |- ... # You can include more modality in here, but currently we only have T2 MRIs
|  |_ Crohns2023_001_seg.nii.gz # This represents the label data
|- Crohns2023_002
|- ...
```
Note that your image and segmentation should be aligned (in image direction), with a same origin and image spacing (i.e. a matched geometry). For the dataset example I provided above (yours should be similar with mine), I provided the following API to convert a specific dataset to the nnU-Net accepted dataset format:

```bash
# Usage: python crohns/utils/convert_dataset.py -i ORIGINAL-DATASET-PATH --dataset-type TRAIN/VAL/TEST --taks-name YOUR-TASK-NAME --task-id YOUR-UNIQUE-ID
python crohns/utils/convert_dataset.py -i Crohns2023/ --dataset-type train --task-name Crohns2023 --task-id 1
```

The arguments for the scripts have the following meanings:

- The `-i` flag specifies the path of the original dataset.
- The `--dataset-type` flag specifies the type of the provided dataset, it should be one of the **train, val** or **test**.
- The `--task-name` is user-customised, you can just put one unique identifier for your segmentation task.
- The `--task-id` is also user-customised and it is bonded with the value of `--task-name` to identify your converted dataset. Make sure the task id is a unique identifier for your dataset.

For more information, you can refer to [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to learn how to construct a dataset with formats accepted by the nnU-Net, you can also take my code in `nnunet_utils/convert_dataset.py` as a reference.

#### Verify the settings

You can run the following command to verify your converted dataset (I will continue to use the example settings above):

```bash
# USAGE: nnUNetv2_plan_and_preprocess -d TASK-ID --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Successful settings will have the following outputs:

```txt
Fingerprint extraction...
Dataset001_XXXX
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

Plans were saved to /YOUR-PATH/nnUNet_preprocessed/Dataset001_XXXX/nnUNetPlans.json
Preprocessing...
Preprocessing dataset Dataset001_XXXX
Configuration: 2d...
...
Configuration: 3d_fullres...
...
Configuration: 3d_lowres...
INFO: Configuration 3d_lowres not found in plans file nnUNetPlans.json of dataset Dataset001_BraTS2020. Skipping.
```

The `3d_lowres` will be sometimes omitted due to the small image size in the provided dataset. Once you get the output above, you are free for training.

### Training

Please enter the following code for training:

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -device DEVICE --npz
```

For instance, currently we have *Datset001_XXXX* as the `DATASET_NAME`, the **1** as the `DATASET_ID`, and the `UNET_CONFIGURATION` is **2d**. In default the nnUNet is performing a 5-fold cross validation. The `--npz` flag is used for finding the best con figuration for the current configuration. Suppose you want to leave the fold 0 as the validation set, you can type:

```bash
nnUNetv2_train Dataset001_XXXX 2d 0 -device cuda --npz
```

You should train all the folds (0 - 4) if you wan to find the best configuration with ensembling. However, if you are training with a proxy model, you can specify your `FOLD` argument as `all`, i.e.

```bash
nnUNetv2_train Dataset001_XXXX 2d all -device cuda --npz
```

### Work with pre-trained weights or proxy model
Please refer to [this link](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md) for further information.

### Find best configuration
After training on all folds, you can use the following command for finding the best configuration with a specific settings, e.g. `2d`, `3d_fullres` etc.
```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c UNET_CONFIGURATION
```

### Inference
Once the best configuration is found, you can use the following command for inference on the unseen data

```bash
nnUNetv2_predict -d DATASET_NAME_OR_ID -i PATH_TO_UNSEEN_DATA -o PATH_TO_OUTPUT_PREDICTIONS -f  0 1 2 3 4 -c UNET_CONFIGURATION
```
The command may change due to your settings, but you can always check the [official documentation](https://github.com/MIC-DKFZ/nnUNet#readme) for detailed explation.

### Postprocessing
You might need postprocessing after making inference, see [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#apply-postprocessing) for more information.

Enjoy your journey in this project!
