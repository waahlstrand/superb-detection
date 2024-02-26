import sklearn.metrics
import sklearn.preprocessing
import numpy as np
from typing import *
from pathlib import Path
import pandas as pd

def grouped_classes(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Tuple[List[int], List[int]],
        n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Group the true and predicted labels according to the provided groups.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples)
        groups: Tuple of lists of class indices e.g

    Returns:
        Tuple of grouped true and predicted labels
    """
    assert len(groups) == 2, "groups must be a tuple of two lists"
    assert len(groups[0]) + len(groups[1]) == n_classes, "The sum of the lengths of the two lists in groups must be equal to n_classes"
    assert set(groups[0] + groups[1]) == set(range(n_classes)), "The union of the two lists in groups must be equal to the set of all class indices"
    
    # Binarize the true labels for each group. If the 
    # true label is in the first group, the binarized label is 1,
    # otherwise it is 0.
    y_true_binary = np.zeros((y_true.shape[0]))
    for i, group in enumerate(groups):
        y_true_binary = np.where(np.isin(y_true, group), i, y_true_binary)

    # Sum the predicted probabilities for each group
    y_pred_grouped = np.zeros((y_pred.shape[0], 2))

    for i, group in enumerate(groups):
        y_pred_grouped[:, i] = y_pred[:, group].sum(axis=1)

    y_pred_grouped = 1-y_pred_grouped[:,0]

    return y_true_binary, y_pred_grouped

def grouped_roc_ovr(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Tuple[List[int], List[int]],
        n_classes: int,
) -> Dict[Literal["fpr", "tpr", "thresholds", "roc_auc", "youden_threshold"], np.ndarray]:
    
    """
    Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy,
    grouping the classes according to the provided groups.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples, n_classes)
        groups: Tuple of lists of class indices e.g. ([0, 1], [2, 3])
        n_classes: Number of classes
        label_dict: Dictionary mapping class indices to class names
    """
    y_true_binary, y_pred_grouped = grouped_classes(y_true, y_pred, groups, n_classes)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true_binary, y_pred_grouped)
    auc = sklearn.metrics.auc(fpr, tpr)

    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]


    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": auc,
        "youden_threshold": youden_threshold
    }

def roc_ovr(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_classes: int, 
        label_dict: Dict[int, str],
        ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute ROC curve for a multi-class classification problem using the One-vs-Rest (OvR) strategy.

    Args:
        y_true: True labels (n_samples)
        y_pred: Predicted labels (n_samples, n_classes)
        n_classes: Number of classes
        label_dict: Dictionary mapping class indices to class names
        path: Path to save the ROC curve plot

    Returns:

    """

    lb = sklearn.preprocessing.LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = sklearn.metrics.roc_curve(y_true_binary[:, i], y_pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    return {
        label_dict[i]: {
            "fpr": fpr[i],
            "tpr": tpr[i],
            "thresholds": thresholds[i],
            "roc_auc": roc_auc[i],
        }
        for i in range(n_classes)
    }

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        label_dict: Dictionary mapping class indices to class names

    Returns:
        np.ndarray: Confusion matrix
    """
    return sklearn.metrics.confusion_matrix(y_true, y_pred, **kwargs)
