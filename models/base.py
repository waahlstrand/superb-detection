from typing import *
from data.types import Batch, Output
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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

        self.test_true = []
        self.test_pred = []


    def forward(self, x: Tensor) -> Output:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Output:
        return super().__call__(*args, **kwds)
    
    def step(self, batch: Batch, batch_idx: int, name: str = "", **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:

        loss, output = self.step(batch, batch_idx, name="train_stage", prog_bar=False, on_step=True, on_epoch=True)
        
        return {
            "loss": loss,
            **output
            # **{f"pred_{k}": v for k, v in output.items()},
        }
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="val_stage", prog_bar=False, on_step=False, on_epoch=True)
       
       return {
            "loss": loss,
            **output
            # **{f"pred_{k}": v for k, v in output.items()},
        }
    
    def on_validation_epoch_end(self) -> None:
        pass

    
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:

       loss, output = self.step(batch, batch_idx, name="test_stage", prog_bar=False, on_step=False, on_epoch=True)

       x, y = batch.x, batch.y
       sizes  = batch.original_sizes

       # Format the output to be compatible with the postprocessor
    #    output = {f"pred_{k}": v for k, v in output.items()}
       y_pred = self.postprocessors["bbox"](output, torch.tensor(sizes, device=output["pred_boxes"].device))

       # Format the target to be compatible with the postprocessor
       y_true = [_.to_dict() for _ in y]

       self.test_true.extend(y_true)
       self.test_pred.extend(y_pred)

       return {
            "loss": loss,
            **output
        }
    
    def on_test_epoch_end(self) -> None:
        
        
        mAP = self.metrics["test_stage"]["map"](self.test_pred, self.test_true)

        self.log_dict({f"test_stage/{k}": v for k, v in mAP.items()})

        self.test_true = []
        self.test_pred = []

    
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
        optimizer   = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return {
            "optimizer": optimizer,
        }
    
    def build_metrics(self) -> nn.ModuleDict:
            
            raise NotImplementedError