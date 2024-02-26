import sklearn.metrics
import sklearn.preprocessing
import numpy as np
from typing import *
from pathlib import Path
import pandas as pd

def roc_ovr(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_classes: int, 
        label_dict: Dict[int, str],
        path: Optional[Path] = None
        ):

    lb = sklearn.preprocessing.LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = sklearn.metrics.roc_curve(y_true_binary[:, i], y_pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    if path is not None:


