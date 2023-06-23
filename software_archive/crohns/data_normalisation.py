import shutil

import numpy as np
import SimpleITK as sitk


def correct_geometry(img_path, label_path):
    """Correct the geometry of the image and the label.

    Args:
        img_path (Path): Path to the image.
        label_path (Path): Path to the label.
    """

    img = sitk.ReadImage(str(img_path))
    label = sitk.ReadImage(str(label_path))

    if not compare_geometry(img_path, label_path):
        # correct the label direction using resample
        label.SetOrigin((0.0, 0.0, 0.0))
        label.SetSpacing((1.0, 1.0, 1.0))

        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetSpacing((1.0, 1.0, 1.0))

        img_direction = img.GetDirection()
        resample_transform = sitk.Transform(3, sitk.sitkIdentity)
        resample_transform.SetParameters(img_direction)
        # resample the label
        label = sitk.Resample(
            label,
            img,
            resample_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            label.GetPixelID(),
        )
        # write the label
        sitk.WriteImage(label, str(label_path))
        # write the image
        sitk.WriteImage(img, str(img_path))


def compare_geometry(img_path, label_path):
    """Compare the geometry of the image and the label.

    Args:
        img_path (Path): Path to the image.
        label_path (Path): Path to the label.
    """

    img = sitk.ReadImage(str(img_path))
    label = sitk.ReadImage(str(label_path))

    img_origin = img.GetOrigin()
    label_origin = label.GetOrigin()

    img_spacing = img.GetSpacing()
    label_spacing = label.GetSpacing()

    img_direction = img.GetDirection()
    label_direction = label.GetDirection()

    img_size = img.GetSize()
    label_size = label.GetSize()

    match = True

    if not np.allclose(img_origin, label_origin):
        print("Origins do not match")
        print("Image origin: ", img_origin)
        print("Label origin: ", label_origin)
        match = False

    if not np.allclose(img_spacing, label_spacing):
        print("Spacings do not match")
        print("Image spacing: ", img_spacing)
        print("Label spacing: ", label_spacing)
        match = False

    if not np.allclose(img_direction, label_direction):
        print("Directions do not match")
        print("Image direction: ", img_direction)
        print("Label direction: ", label_direction)
        match = False

    if not np.allclose(img_size, label_size):
        print("Sizes do not match")
        print("Image size: ", img_size)
        print("Label size: ", label_size)
        match = False

    if not match:
        print("Image: ", img_path.name)

    return match


def convert_to_binary_mask(label_folder_path, output_folder_path):
    """Convert the labels to binary masks and save them in the output folder.

    Args:
        label_folder_path (Path): Path to the folder containing the labels.
        output_folder_path (Path): Path to the folder where the binary masks will be saved.
    """

    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)
    unique_values_dict = {}
    for label_path in sorted(label_folder_path.glob("*.nii.gz")):
        # read the image
        sitk_coronal_label = sitk.ReadImage(str(label_path))
        # convert to numpy array
        data = sitk.GetArrayFromImage(sitk_coronal_label)

        key = tuple(np.unique(data))
        if key not in unique_values_dict:
            unique_values_dict[key] = []

        unique_values_dict[key].append(label_path)

    for keys, values in unique_values_dict.items():
        if keys != (0.0, 1.0):
            # rewrite any label > 0 to 1
            for label_path in values:
                print("Rewriting label: ", label_path)
                sitk_coronal_label = sitk.ReadImage(str(label_path))
                data = sitk.GetArrayFromImage(sitk_coronal_label)
                data[data > 0] = 1
                sitk_coronal_label = sitk.GetImageFromArray(data)
                sitk_coronal_label.CopyInformation(
                    sitk.ReadImage(str(label_path))
                )
                sitk.WriteImage(
                    sitk_coronal_label,
                    str(output_folder_path / label_path.name),
                )
        else:
            # copy the label as is
            for label_path in values:
                shutil.copy(
                    str(label_path), str(output_folder_path / label_path.name)
                )
