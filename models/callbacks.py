from typing import *
from typing import Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from rich import print
import wandb
from data.types import Batch
from data.constants import VERTEBRA_NAMES

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

    def on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: Batch, batch_idx: int, name: str) -> None:
        
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

        f, ax = plt.subplots(
            nrows=1,
            ncols=self.n_samples,
            squeeze=True,
            dpi=300,
        )

        # Get image
        x, y = batch.x, batch.y

        x = x.detach().cpu().squeeze()

        prob = F.softmax(outputs["logits"], -1)

        # print(prob.shape)


        scores, labels = prob[..., :-1].max(-1)

        # print(labels.unique())

        # Select random samples
        idxs = torch.randperm(x.shape[0])[:self.n_samples].detach().cpu()

        # Plot image
        for i in range(self.n_samples):

            idx = idxs[i]
            image_height, image_width = x[idx].shape
            ax[i].imshow(x[idx], cmap="gray", aspect="auto")


            if "keypoints" in outputs and self.plot_keypoints:

                # Get ground truth
                keypoints   = y[idx].keypoints.reshape(-1, 2).detach().cpu()

                # Get predictions
                pred_keypoints = outputs["keypoints"][idx].reshape(-1, 2).detach().cpu()

                # Unnormalize keypoints
                keypoints[:, 0] *= image_width
                keypoints[:, 1] *= image_height

                pred_keypoints[:, 0] *= image_width
                pred_keypoints[:, 1] *= image_height

                # Plot keypoints
                
                ax[i].scatter(keypoints[:, 0], keypoints[:, 1], s=2, color="green", label="ground truth")

                # Chunk the pred_keypoints into groups of 6
                pred_keypoints = pred_keypoints.reshape(-1, 6, 2)

                for j in range(pred_keypoints.shape[0]):
                    
                    # Add label and score
                    label_idx = labels[idx][j].item()
                    score = scores[idx][j].item()

                    if label_idx == self.n_classes:
                        continue
                    else:
                        ax[i].scatter(pred_keypoints[j, :, 0], pred_keypoints[j, :, 1], s=2, color="red")#, label="predictions")
                
                
            if "bboxes" in outputs and self.plot_bboxes:
                boxes = y[idx].boxes.reshape(-1, 4).detach().cpu()
                pred_boxes = outputs["bboxes"][idx].reshape(-1, 4).detach().cpu()

                # Plot boxes
                for box in boxes:
                    cx, cy, w, h = box
                    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                    x1, y1, x2, y2 = x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height
                    ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", alpha=0.5))

                for j, box in enumerate(pred_boxes):

                    # Add label and score
                    label_idx = labels[idx][j].item()
                    score = scores[idx][j].item()

                    if label_idx == self.n_classes:
                        continue

                    else:
                        label = VERTEBRA_NAMES[label_idx]
                        cx, cy, w, h = box
                        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                        x1, y1, x2, y2 = x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height
                        ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", alpha=0.5))



                        ax[i].text(x1, y1, f"{label}", color="white", fontsize=3)


        

            # Remove axes and ticks
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].axis("off")

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.legend()

        if self.save_to_disk:
            f.savefig(f"plot_test.png")

        return f, ax
    
    # def plot(self, outputs: Dict[str, Tensor], batch: Batch) -> Tuple[plt.Figure, plt.Axes]:

    #     f, ax = plt.subplots(
    #         nrows=1,
    #         ncols=self.n_samples,
    #         squeeze=True,
    #         dpi=300,
    #     )

    #     # Get image
    #     x, y = batch.x, batch.y

    #     x = x.detach().cpu().squeeze()

    #     prob = F.softmax(outputs["logits"], -1)
    #     # print(prob.shape)

    #     # Shape (batch_size, n_queries, n_classes + 1)
    #     # Get the highest scoring query for each class
        
    #     scores, object = prob.max(1)

    #     # print(labels.unique())

    #     # Select random samples
    #     idxs = torch.randperm(x.shape[0])[:self.n_samples].detach().cpu()

    #     # Plot image
    #     for i in range(self.n_samples):

    #         idx = idxs[i]
    #         image_height, image_width = x[idx].shape
    #         ax[i].imshow(x[idx], cmap="gray", aspect="auto")

    #         # Get ground truth
    #         if "keypoints" in outputs:

    #             # Get ground truth
    #             keypoints   = y[idx].keypoints.reshape(-1, 2).detach().cpu()

    #             # Unnormalize keypoints
    #             keypoints[:, 0] *= image_width
    #             keypoints[:, 1] *= image_height

    #             ax[i].scatter(keypoints[:, 0], keypoints[:, 1], s=2, color="green", label="ground truth")

    #             # Get predictions
    #             pred_keypoints = outputs["keypoints"][idx].reshape(-1, 2).detach().cpu()

    #             pred_keypoints[:, 0] *= image_width
    #             pred_keypoints[:, 1] *= image_height    

    #             # Reshape into (n_queries, n_keypoints, 2)
    #             # print(pred_keypoints.shape)
    #             pred_keypoints = pred_keypoints.reshape(-1, 6, 2) 

    #         for label_idx, j in enumerate(range(object.shape[1])):

    #             # Add label and score
    #             # label_idx = object[idx][j].item()
    #             score = scores[idx][j].item()

    #             ax[i].scatter(pred_keypoints[j, :, 0], pred_keypoints[j, :, 1], s=2, color="red", marker="o")


    #         if "bboxes" in outputs:

    #             # Get ground truth
    #             boxes = y[idx].boxes.reshape(-1, 4).detach().cpu()

    #             # Plot boxes
    #             for box in boxes:
    #                 cx, cy, w, h = box
    #                 x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    #                 x1, y1, x2, y2 = x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height
    #                 ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", alpha=0.5))

    #             # Get predictions
    #             pred_boxes = outputs["bboxes"][idx].reshape(-1, 4).detach().cpu()

    #             for label_idx, j in enumerate(range(object.shape[1])):

    #                 # print(label_idx, j)

    #                 # Add label and score
    #                 # label_idx = object[idx][j].item()
    #                 score = scores[idx][j].item()
    #                 label = VERTEBRA_NAMES[label_idx] if label_idx < self.n_classes else "background"

                    
    #                 # label = VERTEBRA_NAMES[label_idx]
    #                 cx, cy, w, h = pred_boxes[j]
    #                 x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    #                 x1, y1, x2, y2 = x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height
    #                 ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", alpha=0.5))
    #                 ax[i].text(x1, y1, f"{label}", color="white", fontsize=3)


    #         # Remove axes and ticks
    #         ax[i].set_xticks([])
    #         ax[i].set_yticks([])
    #         ax[i].axis("off")

    #     plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #     # plt.legend(loc='outside right lower')

    #     if self.save_to_disk:
    #         f.savefig(f"plot_test.png")

    #     return f, ax