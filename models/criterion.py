from tasks.detection.backbones.realnvp.models import RealNVP
from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.distributions import Normal, Laplace, Beta

from models.backbones.detr.util import box_ops
from models.backbones.detr.util.misc import (nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from models.backbones.detr.models.segmentation import (dice_loss, sigmoid_focal_loss)

class Classifier(nn.Module):

    def __init__(self, tolerance: float = 0.1) -> None:
        super().__init__()

        self.tolerance = tolerance

    @torch.no_grad()
    def forward(self, vertebrae: Tensor) -> Tensor:
        """
        Classify the type of the vertebra depending on its shape.

        Args:
            vertebrae (Tensor): Vertebra to classify, with anterior, middle and posterior points.
            Shape: (B, 6, 2)

        """
        anterior    = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        posterior   = vertebrae[:, 4:6, :]

        # Compute distances between points
        ha = torch.linalg.norm(anterior[:,0,:] - anterior[:,1,:])
        hp = torch.linalg.norm(posterior[:,0,:] - posterior[:,1,:])
        hm = torch.linalg.norm(middle[:,0,:] - middle[:,1,:])

        apr = ha / hp # Anterior/posterior ratio (not used)
        mpr = hp / hm # Middle/posterior ratio
        mar = ha / hm # Middle/anterior ratio

        # Classify the vertebrae
        normal  = (mpr - mar).abs() < self.tolerance
        crush   = ((mpr > 1 + self.tolerance) & (mar < 1 - self.tolerance)) & ~normal
        biconc  = ((mpr > 1 + self.tolerance) & (mar > 1 + self.tolerance)) & ~normal
        wedge   = ((mpr < 1 - self.tolerance) & (mar > 1 + self.tolerance)) & ~normal

        # Create the classification tensor
        classification = torch.stack([crush , biconc, wedge, normal], dim=-1).float()

        return classification



class PolynomialPenalty(nn.Module):

    def __init__(self, 
                 power: int = 3, 
                 n_vertebrae: int = 13, 
                 n_keypoints: int = 6, 
                 n_dims: int = 2) -> None:
        super().__init__()

        self.power = power
        self.n_vertebrae = n_vertebrae
        self.n_keypoints = n_keypoints
        self.n_dims = n_dims


    def compute_polynomial_coefficients(self, x: Tensor, y: Tensor, power: int = 3) -> Tensor:
        """
        Compute the polynomial of degree "power" along the x-axis
        for each vertebra in "vertebrae". The polynomial is computed
        using least squares.

        Args:
            vertebrae (Tensor): [B, ..., 2] tensor of vertebrae coordinates
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

        # Assume the means are distributed along a 
        # polynomial of degree "self.power"
        means = points.reshape(-1, self.n_vertebrae, self.n_keypoints, self.n_dims).mean(dim=-2)

        # Add anchor points at the ends of the image
        # points = torch.tensor([[[0, 0.5], [1, 0.5]]]).repeat(means.shape[0], 1, 1)
        # means = torch.cat([means, points], dim=1)

        y = means[:,:,0]
        x = means[:,:,1]

        # Compute the polynomial coefficients
        coeffs = self.compute_polynomial_coefficients(x, y, self.power)

        # Compute the polynomial in the points
        poly = torch.stack([x**i for i in range(0, self.power+1)], dim=-1)

        # Compute the polynomial in the points
        y_hat = torch.einsum("bik,bk->bi", poly, coeffs)

        # Compute the mean squared error
        mse = (y_hat - y).square().mean()

        return mse


class RLELoss(nn.Module):

    def __init__(self, prior: str = "gaussian", custom: bool = True) -> None:
        
        super().__init__()
        self.eps = 1e-9
        self.flow = RealNVP()
        self.prior = prior
        self.custom = custom

    def forward(self, mu: Tensor, sigma: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            x_hat: (B*N, 2*K*D)
            x: (B*N, K*D)
            
        Returns:
            loss: (B, K, 2)
        """

        BN, KD = mu.shape
        D = 2

        # Calculate the deviation from a sample x
        error = (mu - x) / (sigma + self.eps) # (B*N, K*D)

        # (B*N, K*D)
        log_phi = self.flow.log_prob(error.view(-1, D)).view(BN, -1, 1)

        # (B*N, K*D)
        if self.custom:
            if self.prior == "gaussian":
                log_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2
            elif self.prior == "laplace":
                log_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                raise NotImplementedError
            
        else:
            if self.prior == "gaussian":
                log_q = Normal(mu, sigma).log_prob(x)
            elif self.prior == "laplace":
                log_q = Laplace(mu, sigma).log_prob(x)
            elif self.prior == "beta":
                alpha = 1 + sigma * mu
                beta  = 1 + sigma * (1 - mu)
                log_q = Beta(alpha, beta).log_prob(x)
            else:
                raise NotImplementedError

        sigma = sigma.view(BN, KD//D, D, )
        log_q = log_q.view(BN, KD//D, D, )

        # (B*N, ) by broadcasting (possibly incorrect)
        loss = torch.log(sigma) - log_phi + log_q

        return loss.mean()
    

class VertebraCriterion(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()

        self.keypoint_loss          = RLELoss()
        self.polynomial_penalty     = PolynomialPenalty(power=args.power)

    def forward(self, mu: Tensor, sigma: Tensor, x: Tensor) -> Tensor:

        loss = self.keypoint_loss(mu, sigma, x) + \
               self.mean_deviation_penalty(mu) 

        return loss
    

import torch
import torch.nn.functional as F
from torch import nn


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.eps = 1e-9
        self.register_buffer('empty_weight', empty_weight)
        self.flow = RealNVP()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

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

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # Implement the RLE modifications (assuming Laplace distribution)
        src_sigma = outputs['pred_sigma_boxes'][idx]
        error = loss_bbox / (src_sigma + self.eps)
        log_phi = self.flow.log_prob(error.view(-1, 2)).view(-1, 2).repeat(1, 2)
        log_q = torch.log(src_sigma * 2) + error.square() / 2
        # print(log_phi.shape, log_q.shape, src_sigma.shape, error.shape)
        loss_bbox = torch.log(src_sigma) - log_phi + log_q

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the keypoints, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "keypoints" containing a tensor of dim [nb_target_boxes, 6]
           The target keypoints are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx]
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_keypoints = F.l1_loss(src_keypoints, target_keypoints, reduction='none')

        # Implement the RLE modifications (assuming Laplace distribution)
        src_sigma = outputs['pred_sigma_keypoints'][idx]
        error = loss_keypoints / (src_sigma + self.eps)
        log_phi = self.flow.log_prob(error.view(-1, 2)).view(-1, 6).repeat(1, 2)
        log_q = torch.log(src_sigma * 2) + error.square() / 2
        # print(log_phi.shape, log_q.shape, src_sigma.shape, error.shape)
        loss_keypoints = torch.log(src_sigma) - log_phi + log_q

    

        losses = {}
        losses['loss_keypoints'] = loss_keypoints.sum() / num_boxes

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
            'keypoints': self.loss_keypoints,
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

