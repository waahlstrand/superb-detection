from typing import *

from torch.nn.modules import ModuleDict
from data.types import Batch, Output, Loss, Prediction
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import torchvision.models as models
from dataclasses import asdict
from models.base import Detector
from models.backbones.DINO.util.misc import NestedTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.dino import build_dino

def build_model(args, class_weights: List[float] = None):

    if args.model == "spine-dino":
                    
        model = SpineDINO(args, n_classes = args.n_classes, class_weights=class_weights)

    else:
        raise NotImplementedError
    
    return model

class SpineDINO(Detector):

    def __init__(self, 
                 args, 
                 n_classes: int = 4, 
                 n_keypoints: int = 6, 
                 n_dim: int = 2, 
                 n_channels: int = 3,
                 class_weights: List[float] = None,
                 **kwargs) -> None:

        self.n_keypoints = n_keypoints
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.class_weights = class_weights

        num_queries = args.num_queries
        
        args.num_queries = 100
        args.num_classes = n_classes
        # detector, criterion, postprocessors = build_detr(args, class_weights=class_weights)

        model, criterion, postprocessors = build_dino(args)

        super().__init__(args.lr, args.lr_backbone, args.weight_decay, **kwargs)

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors


    def forward(self, x: NestedTensor, y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:

        # x = self.model(x) # Dict[str, Tensor]
        x = self.model(x, y)

        return x

    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Output]:

        x, y = batch.x, batch.y
        y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels", "indices", "weights"]} for _ in y]

        output = self(x, y)

        # Convert the targets to the format expected by the criterion
        # y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels", "indices"]} for _ in y]

        loss_dict = self.criterion(output, y)

        # Logging
        losses = [
            "loss_ce",
            "loss_bbox",
            "loss_giou",
            "loss_polynomial",
            "class_error",
            "cardinality_error",
            # "auroc",
            # "f1_score",
            # "accuracy",
        ]
        for loss_name in losses:

            self.log(f"{name}/{loss_name}", loss_dict[loss_name], **kwargs)

        weight_dict = self.criterion.weight_dict

        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        self.log(f"{name}/loss", total, **kwargs)

        return total, output
    
    def build_metrics(self) -> ModuleDict:
        
        return ModuleDict({
            "map": MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        })