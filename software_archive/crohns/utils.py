import xml.etree.ElementTree as ET

import numpy as np
from matplotlib import pyplot as plt


def get_centerline_points(centerline_path, TI_centerline_ratio=0.2):
    """Get the centerline points from the centerline XML file.

    Args:
        centerline_path: the path to the centerline XML file.

    Returns:
        a Numpy array of shape (N, 3) where N is the number of centerline points.
        Each row is a centerline point and the columns are the x, y, z coordinates.
    """
    crohns_centerline = ET.parse(centerline_path)
    root = crohns_centerline.getroot()
    centerline_points = []
    for path in root:
        if "name" not in path.attrib:
            continue

        for point in path:
            centerline_points.append(
                (
                    int(point.attrib["x"]),
                    int(point.attrib["y"]),
                    int(point.attrib["z"]),
                )
            )

    crohns_centerline_size = int(len(centerline_points) * TI_centerline_ratio)
    return np.array(centerline_points[:crohns_centerline_size])


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
        the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def get_bbox_from_centerline(centerline_points):
    """Get the bounding box from the centerline points.

    Args:
        centerline_points: a Numpy array of shape (N, 3) where N is the number of
            centerline points. Each row is a centerline point and the columns are
            the x, y, z coordinates.

    Returns:
        a tuple of two tuples. The first tuple is the top left corner of the
        bounding box and the second tuple is the size of the bounding box.
    """
    min_x = np.min(centerline_points[:, 0])
    max_x = np.max(centerline_points[:, 0])
    min_y = np.min(centerline_points[:, 1])
    max_y = np.max(centerline_points[:, 1])
    min_z = np.min(centerline_points[:, 2])
    max_z = np.max(centerline_points[:, 2])

    # the top left corner of the bounding box
    index = (int(min_x), int(min_y), int(min_z))
    # the size of the bounding box
    size = (
        int(max_x - min_x + 1),
        int(max_y - min_y + 1),
        int(max_z - min_z + 1),
    )
    return index, size


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
    """Show the mask on the given axis.

    Args:
        mask: 3-dim Numpy array of type int contiaining 0 or 1. The mask to show.
        ax: the axis to show the mask on.
    """
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Show the bounding box on the given axis.

    Args:
        box: a tuple of four integers, the first two are the top left corner coordinates
            and the last two are the bottom right corner coordinates.
        ax: the axis to show the bounding box on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2
        )
    )
