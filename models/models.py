from typing import *
from data.types import *
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import SpearmanCorrCoef, MeanAbsolutePercentageError, MetricCollection, MeanSquaredError
import lightning as L
import torchvision.models as models
from models.backbones.detr.models.detr import build
from models.criterion import RLELoss
from dataclasses import asdict
from models.backbones.detr.util.misc import NestedTensor
from models.criterion import PolynomialPenalty, SetCriterion
from models.backbones.detr.models.detr import build_backbone, build_transformer, build_matcher
from models.backbones.detr.models.detr import DETR, PostProcess

class Classifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # self.tolerance = tolerance

    @torch.no_grad()
    def forward(self, vertebrae: Tensor , tolerance: float = 0.1) -> Tensor:
        """
        Classify the type of the vertebra depending on its shape.

        Args:
            vertebrae (Tensor): Vertebra to classify, with anterior, middle and posterior points.
            Shape: (B, 6, 2)

        """
        posterior   = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        anterior    = vertebrae[:, 4:6, :]

        # Compute distances between points
        ha = (anterior[:,0,:] - anterior[:,1,:]).norm(dim=-1)
        hp = (posterior[:,0,:] - posterior[:,1,:]).norm(dim=-1)
        hm = (middle[:,0,:] - middle[:,1,:]).norm(dim=-1)

        apr = ha / hp # Anterior/posterior ratio (not used)
        mpr = hp / hm # Middle/posterior ratio
        mar = ha / hm # Middle/anterior ratio

        # Classify the vertebrae
        normal  = (mar <= 1 + tolerance) \
                        & (mar >= 1 - tolerance) \
                        & (mpr <= 1 + tolerance) \
                        & (mpr >= 1 - tolerance) \
                        & (apr <= 1 + tolerance) \
                        & (apr >= 1 - tolerance)

        crush       = ((mpr >= 1 ) & (mar <= 1 )) & (apr < 1)   & ~normal
        biconcave   = ((mpr >= 1 ) & (mar >= 1 ))               & ~normal & ~crush
        wedge       = ((mpr <= 1 ) & (mar >= 1 )) & (apr > 1)   & ~normal & ~crush & ~biconcave
        biconvex    = ((mpr <= 1 ) & (mar <= 1 ))               & ~normal & ~crush & ~biconcave & ~wedge

        # Set biconvex as normal
        normal = normal | biconvex

        # Create the classification tensor
        classification = torch.stack([normal, wedge, biconcave, crush], dim=-1)

        return classification

class FuzzyClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, vertebrae: Tensor , tolerance: float = 0.1) -> Tensor:

        threshold   = 1

        posterior   = vertebrae[:, 0:2, :]
        middle      = vertebrae[:, 2:4, :]
        anterior    = vertebrae[:, 4:6, :]

        ha = (anterior[:,0,:] - anterior[:,1,:]).norm(dim=-1)
        hp = (posterior[:,0,:] - posterior[:,1,:]).norm(dim=-1)
        hm = (middle[:,0,:] - middle[:,1,:]).norm(dim=-1)

        apr = ha / hp
        mpr = hm / hp
        mar = hm / ha

        apr_pos = F.sigmoid((apr - (threshold)))
        mpr_pos = F.sigmoid((mpr - (threshold)))
        mar_pos = F.sigmoid((mar - (threshold)))

        mpr_neg = F.sigmoid((threshold - mpr))
        mar_neg = F.sigmoid((threshold - mar))
        apr_neg = F.sigmoid((threshold - apr))

        normal, ind = torch.stack([
            F.sigmoid((apr - (threshold - tolerance))), 
            F.sigmoid((mpr - (threshold - tolerance))), 
            F.sigmoid((mar - (threshold - tolerance))), 
            F.sigmoid((threshold + tolerance - apr)), 
            F.sigmoid((threshold + tolerance - mpr)), 
            F.sigmoid((threshold + tolerance - mar))
            ], dim=1).min(dim=1)

        crush, ind = torch.stack([
            mpr_pos, 
            mar_neg, 
            apr_pos, 
            1-normal, 
            ], dim=1).min(dim=1)

        biconcave, ind = torch.stack([
            mpr_neg, 
            mar_neg, 
            1-normal, 
            1-crush, 
            ], dim=1).min(dim=1)

        wedge, ind = torch.stack([
            mpr_neg, 
            mar_pos, 
            apr_neg, 
            1-normal, 
            1-crush, 
            1-biconcave
            ], dim=1).min(dim=1)


        classification_probs = torch.stack([normal, wedge, biconcave, crush], dim=-1)

        return classification_probs


class Detector(L.LightningModule):

    def __init__(self, 
                lr: float = 1e-4,
                lr_backbone: float = 1e-5,
                weight_decay: float = 1e-4
                 ) -> None:
        
        super().__init__()

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.metrics  = nn.ModuleDict({
            name: self.build_metrics() for name in ["train_stage", "val_stage", "test_stage"]
        })


    def forward(self, x: Tensor) -> Output:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Output:
        return super().__call__(*args, **kwds)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Loss, Output]:
        raise NotImplementedError
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, output = self.step(batch, batch_idx, name="train_stage", prog_bar=True, on_step=True, on_epoch=True)

        return {
            "loss": loss.total,
            **output.to_dict(),
        }
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="val_stage", prog_bar=False, on_step=False, on_epoch=True)
       
       return {
            "loss": loss.total,
            **output.to_dict(),
        }
    
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="test_stage", prog_bar=False, on_step=False, on_epoch=True)
       
       return {
            "loss": loss.total,
            **output.to_dict(),
        }
    
    def configure_optimizers(self) -> Any:
        
        param_dicts = [
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                  "lr": self.lr,
              },
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        # params = [p for n, p in self.named_parameters() if p.requires_grad]
        # optimizer = torch.optim.Adam(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        optimizer   = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=self.lr*10, step_size_up=10, cycle_momentum=False)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }
    
    def build_metrics(self) -> MetricCollection:
            
            raise NotImplementedError
    

class VertebraeDetector(Detector):

    def __init__(self, 
                 n_vertebrae: int = 13,
                 n_keypoints: int = 6,
                 n_dims: int = 2,
                 backbone: models.ResNet = models.resnet50(weights=None),
                 criterion = RLELoss,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_vertebrae    = n_vertebrae
        self.n_keypoints    = n_keypoints
        self.n_dims         = n_dims
        self.criterion      = criterion(prior="laplace", custom=True)
        
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.n_hidden = backbone.fc.in_features

        self.n_channels     = 3
        self.projection     = nn.Conv2d(1, self.n_channels, kernel_size=1)
        self.embedding      = nn.Linear(n_vertebrae, self.n_hidden)

        # self.bbox           = MLP(self.n_hidden, self.n_vertebrae * self.n_dims * 2)
        self.keypoints      = MLP(self.n_hidden, self.n_vertebrae * self.n_keypoints * self.n_dims)

        # self.sigma_bbox     = MLP(self.n_hidden, self.n_vertebrae * self.n_dims * 2)
        self.sigma_keypoints= MLP(self.n_hidden, self.n_vertebrae * self.n_keypoints * self.n_dims)


    def forward(self, x: Tensor, idx: Tensor) -> Output:
        """
        Args:
            x (Tensor): Image batch of shape [B, 1, H, W]
            idx (Tensor): One-hot vector batch of shape [B, N]
        """

        x = self.projection(x) # [B, 3, H, W]
        embedding = self.embedding(idx.type(torch.double)) # [B, n_hidden, N]

        # Extract features
        features = self.backbone(x).squeeze() # [B, n_hidden, 1, 1]

        # Merge features and embedding
        features = features + embedding # [B, n_hidden, N]

        # Predictions
        keypoints_prediction = Prediction(
            mu=self.keypoints(features).sigmoid(),
            sigma=self.sigma_keypoints(features).sigmoid(),
        )

        # bbox_prediction = Prediction(
        #     mu=self.bbox(features).sigmoid(),
        #     sigma=self.sigma_bbox(features).sigmoid(),
        # )

        return Output(
            # bboxes=bbox_prediction,
            keypoints=keypoints_prediction,
        )
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Loss, Output]:

        x, y = batch.x, batch.y
        idx       = torch.stack([_.index for _ in y]) # [B, N]
        keypoints = torch.cat([_.keypoints for _ in y]) # [B*n_vertebrae, n_keypoints*n_dims]
        # bboxes    = torch.cat([_.bboxes for _ in y]) # [B*n_vertebrae, n_dims*2]

        output = self(x, idx)

        mu_keypoints    = self.filter(output.keypoints.mu, idx)
        # mu_bboxes       = self.filter(output.bboxes.mu, idx)

        sigma_keypoints = self.filter(output.keypoints.sigma, idx)
        # sigma_bboxes    = self.filter(output.bboxes.sigma, idx)

        # loss_bbox       = self.criterion(mu_bboxes, sigma_bboxes, bboxes)
        loss_keypoints  = self.criterion(mu_keypoints, sigma_keypoints, keypoints)

        loss = Loss(
            # bboxes=loss_bbox,
            keypoints=loss_keypoints
                    )

        # Logging
        for loss_name, loss_value in loss.to_dict().items():
            # print(name, loss_name, loss_value)
            self.log(f"{name}_{loss_name}", loss_value, **kwargs)


        return loss, output
    
    def filter(self, x: Tensor, idx: Tensor) -> Tensor:
        """
        Filter a tensor of shape [B, N, ...] to [B*n_vertebrae, ...] using the one-hot vector idx

        Args:
            x (Tensor): Tensor to be filtered of shape [B, N, ...]
            idx (Tensor): One-hot vector of shape [B, N]

        Returns:
            Tensor: Filtered tensor of shape [B*n_vertebrae, ...]
        
        """
        B, N = idx.shape

        x = x.reshape(B*N, -1)
        idx = idx.reshape(B*N)

        return x[idx]
    
    def build_metrics(self) -> MetricCollection:

        metrics = MetricCollection({
            'mse': MeanSquaredError(),
            'mape': MeanAbsolutePercentageError(),
            'spearman': SpearmanCorrCoef(),
        })

        return metrics


class MLP(nn.Module):

    def __init__(self, 
                 n_in: int, 
                 n_out: int, 
                 n_hidden: int = 512, 
                 n_layers: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_in       = n_in
        self.n_out      = n_out
        self.n_hidden   = n_hidden
        self.n_layers   = n_layers

        h = [n_hidden] * (n_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([n_in] + h, h + [n_out]))

    def forward(self, x: Tensor) -> Tensor:

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.n_layers - 1 else layer(x)
        return x
    
    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return super().__call__(*args, **kwds)
    

class VertebraViDT(Detector):

    def __init__(self, 
                 args, 
                 n_classes=4, 
                 n_keypoints: int = 6, 
                 n_dim: int = 2, 
                 n_channels: int = 3,
                 **kwargs) -> None:

        self.n_keypoints = n_keypoints
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.n_channels = n_channels


        num_queries = args.num_queries
        
        args.num_queries = 100
        detector, criterion, postprocessors = build_detr(args)

        super().__init__(args.lr, args.lr_backbone, args.weight_decay, **kwargs)
        
        # self.projection     = nn.Conv2d(1, self.n_channels, kernel_size=1)
        # self.model  = nn.Sequential(self.projection, detector)
        self.model = detector

        
        self.model.class_embed = nn.Linear(self.model.class_embed.in_features, 91 + 1)

        if args.frozen_weights:
            checkpoint = torch.load(args.frozen_weights, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.model.eval()
            # for param in self.model.parameters():
            #     param.requires_grad = False
        # print(num_queries, args.num_queries)
        if num_queries != 100:
            self.model.query_embed = nn.Embedding(num_queries, self.model.transformer.d_model)

        self.model.class_embed = nn.Linear(self.model.class_embed.in_features, n_classes + 1)

        try:
            torch.compile(self.model, mode="reduce-overhead")
        except:
            raise ValueError("The model is not compilable.")


        self.criterion = criterion
        self.postprocessors = postprocessors
        self.poly_weight = args.polynomial_loss_coef
        self.polynomial_penalty = PolynomialPenalty(power=3, n_vertebrae=num_queries)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:

        mask = (x == -1).squeeze()
        x    = x.repeat(1, 3, 1, 1)
        
        nested_tensor = NestedTensor(x, mask)


        x = self.model(nested_tensor) # Dict[str, Tensor]

        # keypoints_prediction = Prediction(
        #     mu=x["keypoints"].sigmoid(),
        #     sigma=None,
        # )

        # bbox_prediction = Prediction(
        #     mu=x["boxes"].sigmoid(),
        #     sigma=None,
        # )

        # return Output(
        #     bboxes=bbox_prediction,
        #     keypoints=keypoints_prediction,
        # )

        return x

    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Output:

        x, y = batch.x, batch.y

        output = self(x)

        # Convert the targets to the format expected by the criterion
        y = [{k: v for k, v in asdict(_).items() if k in ["boxes", "keypoints", "labels"]} for _ in y]

        loss_dict = self.criterion(output, y)
        
        polynomial_penalty = self.poly_weight*self.polynomial_penalty(output["pred_keypoints"])
        
        
        loss = Loss(
            keypoints=self.criterion.weight_dict["loss_keypoints"]*loss_dict["loss_keypoints"],
            boxes=self.criterion.weight_dict["loss_bbox"]*loss_dict["loss_bbox"],
            giou=self.criterion.weight_dict["loss_giou"]*loss_dict["loss_giou"],
            cross_entropy=self.criterion.weight_dict["loss_ce"]*loss_dict["loss_ce"],
            polynomial=polynomial_penalty,
        
        )

        # Logging
        for loss_name, loss_value in loss.to_dict().items():
            # print(name, loss_name, loss_value)
            self.log(f"{name}_{loss_name}", loss_value, **kwargs)


        # Metrics
        # keypoints = torch.cat([_["keypoints"] for _ in y])
        # pred_keypoints = output["pred_keypoints"] # [B, num_queries, num_keypoints, num_dim]
        # pred_logits    = output["pred_logits"] # [B, num_queries, num_classes+1]

        # Select the top 13 queries 

        # for metric_name, metric in self.metrics[name].items():
        #     m = metric(output["pred_keypoints"].view(-1, self.n_keypoints*self.n_dim), keypoints)
        #     self.log(f"{name}_{metric_name}", m, **kwargs)

        output = Output(
            bboxes=Prediction(mu=output["pred_boxes"], sigma=None),
            keypoints=Prediction(mu=output["pred_keypoints"], sigma=None),
            logits=output["pred_logits"],
        )

        
        return loss, output

    def build_metrics(self) -> MetricCollection:

        metrics = MetricCollection({
            'mse': MeanSquaredError(),
            'mape': MeanAbsolutePercentageError(),
            'spearman': SpearmanCorrCoef(),
        })

        return metrics



def build_detr(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    elif args.dataset_file == "superb":

        num_classes = args.n_classes

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': 1, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_keypoints': args.keypoint_loss_coef,
        'loss_giou': args.giou_loss_coef,
        }

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'keypoints']

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors



class FlowLikelihood(nn.Module):

    def __init__(self, model: VertebraViDT) -> None:
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:

        batch_size, n_channels, H, W = x.shape

        output = self.model(x)

        mu, sigma = output["pred_keypoints"], output["pred_sigma_keypoints"]

        mu, sigma = mu.view(batch_size, -1, 2), sigma.view(batch_size, -1, 2)

        Y, X = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
        x = torch.stack([X, Y], dim=-1).reshape(-1, 2).to(self.model.device)

        eps = 1e-9
        for idx in range(batch_size):
            error = (mu[idx, :, None] - x) / (sigma[idx, :, None] + eps)

            log_phi = self.model.criterion.flow.log_prob(error.view(-1, 2)).view(-1, 20, 6, 1)
            log_q   = torch.log(2*sigma) + torch.abs(error)
            log_likelihood = (torch.log(sigma) - log_phi + log_q).sum(dim=-1)
            
