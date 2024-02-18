from typing import *
from typing import Any
import torch
from torch import Tensor
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rich import print
import wandb
from data.types import Batch, Output
from models.backbones.DINO.util import box_ops
import torchmetrics as tm
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from models.vertebra import SingleVertebraClassifier

class VertebraePlotCallback(L.Callback):
    
    def __init__(self, 
                    n_samples: int = 4,
                    plot_frequency: int = 100,
                    save_to_disk: bool = False,
                    n_classes: int = 13,
                    plot_keypoints: bool = True,
                    plot_bboxes: bool = True,
                    **kwargs) -> None:
            
        super().__init__(**kwargs)
    
        self.n_samples = n_samples
        self.plot_frequency = plot_frequency
        self.save_to_disk = save_to_disk
        self.n_classes = n_classes
        self.plot_keypoints = plot_keypoints
        self.plot_bboxes = plot_bboxes

    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, name: str) -> None:
        
        if batch_idx % self.plot_frequency == 0:
            f, ax = self.plot(outputs, batch)

            # Log image
            trainer.logger.experiment.log({
                f"{name}_plot": wandb.Image(f, caption=f"{name} plot")
            })

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:

        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")

    def plot(self, outputs: Dict[str, Tensor], batch: Batch) -> Tuple[plt.Figure, plt.Axes]:


        f, ax = plt.subplots(1, 1, squeeze=True, dpi=300)
        offset = -20

        nested, y = batch.x, batch.y

        images  = nested.tensors.detach().cpu().squeeze()
        masks   = nested.mask.detach().cpu().squeeze()
        sizes   = batch.original_sizes

        # Select a single random sample
        idx = torch.randint(0, images.shape[0], (1,)).item()

        image = images[idx][0][~masks[idx]].reshape(*sizes[idx])

        image_height, image_width = sizes[idx]
        tensor_height, tensor_width = image.shape

        # Plot image
        ax.imshow(image, cmap="gray", origin="upper")

        # Get ground truth
        keypoints   = y[idx].keypoints.detach().cpu()
        indicators  = y[idx].indices.reshape(-1).detach().cpu()
        true_labels = y[idx].labels.reshape(-1).detach().cpu()

        # Remove keypoints that should not be plotted
        keypoints   = keypoints.reshape(*indicators.shape, -1)[indicators.to(keypoints.device)].reshape(-1, 2)
        true_labels = true_labels.reshape(*indicators.shape)[indicators.to(true_labels.device)].reshape(-1)

        # Get predictions
        pred_keypoints  = outputs["keypoints"][idx].detach().cpu().reshape(-1, 2)
        pred_labels     = outputs["logits"][idx].detach().cpu().softmax(-1).argmax(-1)

        # Unnormalize keypoints
        keypoints[:, 0] *= image_width
        keypoints[:, 1] *= image_height


        pred_keypoints[:, 0] *= tensor_width
        pred_keypoints[:, 1] *= tensor_height


        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=2, color="green", label="Annotations")
        ax.scatter(pred_keypoints[:, 0], pred_keypoints[:,1], s=3, edgecolors="red", facecolors='none', linewidths=1.0, label="Predictions")

        # Add labels
        keypoints = keypoints.reshape(-1, 6, 2)
        pred_keypoints = pred_keypoints.reshape(-1, 6, 2)
        for k in range(len(keypoints)):
            label = true_labels[k].item()
            ax.text(keypoints[k, 0, 0] + offset, keypoints[k, 0, 1], f"{label}", color="green", fontsize=3)

        for k in range(len(pred_keypoints)):
            label = pred_labels[k].item()
            ax.text(pred_keypoints[k, 0, 0] + offset, pred_keypoints[k, 0, 1], f"{label}", color="red", fontsize=3)


        plt.subplots_adjust(wspace=0.05, hspace=0.05)        
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)

        # if self.save_to_disk:
        # f.savefig(f"plot_test.png", bbox_inches="tight")

        return f, ax
    

class SpinePlotCallback(L.Callback):
    
    def __init__(self, 
                    n_samples: int = 4,
                    plot_frequency: int = 100,
                    save_to_disk: bool = False,
                    n_classes: int = 13,
                    **kwargs) -> None:
            
        super().__init__(**kwargs)
    
        self.n_samples = n_samples
        self.plot_frequency = plot_frequency
        self.save_to_disk = save_to_disk
        self.n_classes = n_classes

    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Batch, batch_idx: int, name: str) -> None:
        
        if batch_idx % self.plot_frequency == 0:
            f, ax = self.plot(outputs, batch, pl_module)

            # Log image
            trainer.logger.experiment.log({
                f"{name}_plot": wandb.Image(f, caption=f"{name} plot")
            })

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Any, batch_idx: int) -> None:

        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")

    def plot(self, outputs: Dict[str, Tensor], batch: Batch, module: L.LightningModule) -> Tuple[plt.Figure, plt.Axes]:

        postprocess = module.postprocessors["bbox"]

        f, ax = plt.subplots(1, 1, squeeze=True, dpi=300)
        offset = -20

        nested, y = batch.x, batch.y

        images  = nested.tensors.detach().cpu().squeeze()
        masks   = nested.mask.detach().cpu().squeeze()
        sizes   = batch.original_sizes

        # Select a single random sample
        idx = torch.randint(0, images.shape[0], (1,)).item()

        image = images[idx][0][~masks[idx]].reshape(*sizes[idx])

        image_height, image_width = sizes[idx]
        tensor_height, tensor_width = image.shape

        # Plot image
        ax.imshow(image, cmap="gray", origin="upper")

        # Get ground truth
        bboxes      = y[idx].boxes.detach().cpu()
        indicators  = y[idx].indices.reshape(-1).detach().cpu()
        true_labels = y[idx].labels.reshape(-1).detach().cpu()

        # Remove keypoints that should not be plotted
        bboxes      = bboxes.reshape(*indicators.shape, -1)[indicators.to(bboxes.device)].reshape(-1, 4)
        true_labels = true_labels.reshape(*indicators.shape)[indicators.to(true_labels.device)].reshape(-1)

        # Unnormalize bboxes
        bboxes = box_ops.box_cxcywh_to_xyxy(bboxes)
        bboxes = bboxes * torch.tensor([image_width, image_height, image_width, image_height], dtype=bboxes.dtype)

        # Get predictions
        pred_bboxes     = outputs["pred_boxes"][idx].detach().cpu()
        processed       = postprocess(outputs, torch.tensor(sizes, device=outputs["pred_boxes"].device)) # List[Dict[str, Tensor]]
        pred_bboxes     = processed[idx]["boxes"].detach().cpu()
        pred_labels     = outputs["pred_logits"][idx].detach().cpu().softmax(-1).argmax(-1)

        # Plot bounding boxes
        for k in range(len(bboxes)):
            label = true_labels[k].item()
            box = bboxes[k]
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", linewidth=1))
            ax.text(x1 + offset, y1, f"{label}", color="green", fontsize=3)

        # print(pred_bboxes.shape, pred_labels.shape)
        for k in range(len(pred_bboxes)):
            label = pred_labels[k].item()
            box = pred_bboxes[k]
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=1))
            ax.text(x1 + offset, y1, f"{label}", color="red", fontsize=3)

        ground_truth = mpatches.Patch(color='green', label='Ground truth')
        predicted = mpatches.Patch(color='red', label='Predicted')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)        
        plt.legend(handles=[ground_truth, predicted], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=1)

        if self.save_to_disk:
            f.savefig(f"plot_test.png", bbox_inches="tight")

        return f, ax

class VertebraPlotCallback(L.Callback):

    def __init__(self, n_samples: int = 4, plot_frequency: int = 10):

        super().__init__()

        self.n_samples = n_samples
        self.plot_frequency = plot_frequency


    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch_idx: int, name: str) -> None:
        
        if batch_idx % self.plot_frequency == 0:
            f, ax = self.plot(outputs, n_samples=self.n_samples)

            # Log image
            trainer.logger.experiment.log({
                f"{name}/plot": wandb.Image(f, caption=f"{name} plot")
            })


    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int) -> None:

        self.on_batch_end(trainer, pl_module, outputs, batch_idx, "train")

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int) -> None:
        
        self.on_batch_end(trainer, pl_module, outputs, batch_idx, "val")

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Dict[str, Tensor], batch: Any, batch_idx: int, *args, **kwargs) -> None:

        images = outputs["images"].detach().cpu() if outputs is not None else []

        for i, image in enumerate(images):
            f, ax = plt.subplots(1, 1, squeeze=True, dpi=300)
            self.plot_image(
                image[0], 
                outputs["keypoints"][i], 
                outputs["grades"][i],
                outputs["types"][i],
                outputs["pred_keypoints"][i],
                outputs["pred_keypoints_sigma"][i],
                outputs["pred_grades"][i],
                outputs["pred_types"][i],
                outputs.get("loss_per_pixel", None), 
                ax)
            trainer.logger.experiment.log({
                f"test_stage/plot_image": wandb.Image(f, caption=f"Test plot image")
            })

            plt.close(f)

    def plot_image(self, 
                   image: Tensor, 
                   keypoints: Tensor, 
                   grades: Tensor, 
                   types: Tensor,
                   pred_keypoints: Tensor, 
                   pred_keypoints_sigma: Tensor, 
                   pred_grades: Tensor, 
                   pred_types: Tensor,
                   loss_per_pixel: Union[Tensor, None], 
                   ax: plt.Axes) -> None:


        image = image.detach().cpu().squeeze()
        keypoints_ = keypoints.detach().cpu().reshape(-1, 2)
        grades_ = grades.detach().cpu()
        types_  = types.detach().cpu()
        pred_keypoints_ = pred_keypoints.detach().cpu().reshape(-1, 2)
        pred_keypoints_sigma_ = pred_keypoints_sigma.detach().cpu().reshape(-1, 2).mean()
        pred_grades_ = pred_grades.detach().cpu().softmax(-1).argmax(-1)
        pred_types_ = pred_types.detach().cpu().softmax(-1).argmax(-1)

        # Unnormalize keypoints
        keypoints_[:, 0] *= image.shape[-1]
        keypoints_[:, 1] *= image.shape[-2]

        pred_keypoints_[:, 0] *= image.shape[-1]
        pred_keypoints_[:, 1] *= image.shape[-2]

        # Plot image
        ax.imshow(image, cmap="gray", origin="upper")

        if loss_per_pixel is not None:
            loss_per_pixel = loss_per_pixel.detach().exp().cpu()
            m = torch.max(loss_per_pixel).numpy()
            # step = 1
            # levels = np.arange(0.0, m, step) + step
            n_levels = 7
            xx, yy = np.meshgrid(np.arange(loss_per_pixel.shape[-2]), np.arange(loss_per_pixel.shape[-1]))
            for k in range(loss_per_pixel.shape[0]):
                ax.contourf(yy, xx, loss_per_pixel[k].t().cpu(), levels=n_levels, cmap="Reds", origin="upper", alpha=0.1)

        # Plot keypoints
        ax.scatter(keypoints_[:, 0], keypoints_[:, 1], s=4, color="green", label="Annotations")
        ax.scatter(pred_keypoints_[:, 0], pred_keypoints_[:,1], s=2, edgecolors="red", facecolors='none', linewidths=1.0, label="Predictions")

        # Add labels
        keypoints_ = keypoints_.reshape(-1, 6, 2)
        pred_keypoints_ = pred_keypoints_.reshape(-1, 6, 2)
            

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"g: {grades_.item()} ({pred_grades_.item()}), t: {types_.item()} ({pred_types_.item()})", fontsize=5)

    def plot(self, outputs: Dict[str, Tensor], n_samples: int): 
        """
        Plots a random sample of vertebrae from the batch on a grid, with keypoints and labels
        """

        # images, y = batch.x, batch.y
        images = outputs["images"].detach().cpu()
        n_samples = self.n_samples if self.n_samples <= images.shape[0] else images.shape[0]

        f, ax = plt.subplots(nrows=1, ncols=n_samples, squeeze=True, dpi=150)
        
        # Select a number of random samples
        idxs = torch.randperm(images.shape[0])[:n_samples]

        # Plot each sample
        for i in range(n_samples):

            idx = idxs[i].item()

            self.plot_image(
                outputs["images"][idx][0], 
                outputs["keypoints"][idx], 
                outputs["grades"][idx],
                outputs["types"][idx], 
                outputs["pred_keypoints"][idx], 
                outputs["pred_keypoints_sigma"][idx], 
                outputs["pred_grades"][idx], 
                outputs["pred_types"][idx], 
                outputs.get("loss_per_pixel", None),
                ax[i]
                )

            
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
        return f, ax

                