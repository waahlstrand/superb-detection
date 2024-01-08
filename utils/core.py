import numpy as np
import torch

def bbox_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Calculates the bounding box from the keypoints in the format (xyxy).

    Args:
        keypoints: Numpy array of shape (batch, n_keypoints, 2)
    """

    x_min = keypoints[:, :, 0].min(axis=1)
    x_max = keypoints[:, :, 0].max(axis=1)
    y_min = keypoints[:, :, 1].min(axis=1)
    y_max = keypoints[:, :, 1].max(axis=1)

    return np.stack((x_min, y_min, x_max, y_max), axis=1)

