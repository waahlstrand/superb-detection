from models.backbones.realnvp.models import RealNVP
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchmetrics

from typing import *

from models.backbones.DINO.util import box_ops
from models.backbones.DINO.util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)

from models.backbones.DINO.models.dino.dino import SetCriterion

class PolynomialPenalty(nn.Module):

    def __init__(self, 
                 power: int = 3, 
                 n_dims: int = 2) -> None:
        super().__init__()

        self.power = power
        self.n_dims = n_dims


    def compute_polynomial_coefficients(self, x: Tensor, y: Tensor, power: int = 3) -> Tensor:
        """
        Compute the polynomial of degree "power" along the x-axis
        for each vertebra in "vertebrae". The polynomial is computed
        using least squares.

        Args:
            vertebrae (Tensor): [B, ..., 4] tensor of vertebrae boxes
            power (int, optional): Degree of polynomial. Defaults to 3.

        Returns:
            Tensor: [B, ..., power+1] tensor of polynomial coefficients
        """
        # Assume that x is distributed along a polynomial of degree 
        # Compute the polynomial
        powers = [x**i for i in range(0, power+1)]
        poly = torch.stack([*powers], dim=-1)

        # Solve the linear system with least squares
        coeffs = torch.linalg.lstsq(poly, y).solution

        return coeffs

    def forward(self, points: Tensor) -> Tensor:
        """
        Args:
            points (Tensor): [B, 4] tensor of bounding boxes"""

        cx, cy, w, h = points.unbind(dim=-1)

        x = cy
        y = cx

        # Compute the polynomial coefficients
        coeffs = self.compute_polynomial_coefficients(x, y, self.power)

        # Compute the polynomial in the points
        poly = torch.stack([x**i for i in range(0, self.power+1)], dim=-1)

        # Compute the polynomial in the points
        # y_hat = torch.einsum("bik,bk->bi", poly, coeffs)
        y_hat  = poly @ coeffs

        # Compute the mean squared error
        mse = (y_hat - y).abs().mean()

        return mse

class SetCriterion(SetCriterion):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)

        self.polynomial_penalty = PolynomialPenalty(power=3, n_dims=2)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_weights = torch.cat([t["weights"][J] for t, (_, J) in zip(targets, indices)]).unsqueeze(-1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') 

        polynomial_penalty = self.polynomial_penalty(src_boxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_polynomial'] = polynomial_penalty
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))) 
        
        loss_giou = loss_giou * target_weights.squeeze(-1)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    
class OldSetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, 
                 num_classes, 
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses, 
                 tolerance: float = 0.2, 
                 n_vertebrae: int = 13,
                 n_dims: int = 2, 
                 missing_weight: float = 1e-3,
                 class_weights: List[float] = []):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        if len(class_weights) == 0:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
        else:
            empty_weight = torch.zeros(len(class_weights) + 1)
            empty_weight[-1] = self.eos_coef
            empty_weight[:-1] = torch.tensor(class_weights)

        self.eps = 1e-9
        self.register_buffer('empty_weight', empty_weight)
        
        self.polynomial_penalty = PolynomialPenalty(power=3, n_dims=n_dims)

        self.n_dims = n_dims
        self.n_vertebrae = n_vertebrae
        self.tolerance = tolerance
        self.missing_weight = missing_weight
    
        # Metrics
        self.auroc      = torchmetrics.AUROC(task="multiclass", num_classes=num_classes+1) #if num_classes > 1 else torchmetrics.AUROC(task="multiclass", num_classes=num_classes+1)
        self.accuracy   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes+1) #if num_classes > 1 else torchmetrics.Accuracy(task="binary")
        self.f1_score   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes+1) #if num_classes > 1 else torchmetrics.F1Score(task="binary")

    def loss_labels(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # idx = self._get_src_permutation_idx(indices) # Tuple of (original index, permuted index)
        # target_classes_o    = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_indicators   = torch.cat([t["indices"][J] for t, (_, J) in zip(targets, indices)])
        # # target_classes_o    = target_classes_o[target_indicators]

        # # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': self.weight_dict['loss_ce'] * loss_ce}

        # Log metrics
        losses["auroc"]     = self.auroc(src_logits.transpose(1, 2).detach(), target_classes.detach())
        losses["accuracy"]  = self.accuracy(src_logits.transpose(1, 2).detach(), target_classes.detach())
        losses["f1_score"]  = self.f1_score(src_logits.transpose(1, 2).detach(), target_classes.detach())

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_indicators   = torch.cat([t["indices"][J] for t, (_, J) in zip(targets, indices)])

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        polynomial_penalty = self.polynomial_penalty(src_boxes)

        # Weight loss per sample if not present
        loss_bbox[~target_indicators] = self.missing_weight*loss_bbox[~target_indicators]

        losses = {}
        losses['loss_polynomial'] = self.weight_dict['loss_polynomial'] * polynomial_penalty
        losses['loss_bbox'] = self.weight_dict['loss_bbox'] * loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = self.weight_dict['loss_giou'] * loss_giou.sum() / num_boxes
        
        return losses
    

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

