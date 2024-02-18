
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import lightning as L
from .superb import SuperbDataset, build_augmenter, SuperbStatistics
from torch import Tensor
import torch.nn as nn
from typing import *
from data.types import *
from torchvision.ops import roi_align
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from sklearn.utils.class_weight import compute_class_weight
from utils.core import as_ordered
import kornia.augmentation as K




def normalize(x: np.ndarray, max_val: float) -> Tensor:

    x_max   = x.max().astype(np.float32)
    max_val = torch.max(torch.tensor(max_val), torch.tensor(x_max)).numpy()

    x = torch.from_numpy(x.copy() / max_val).type(torch.float32)

    return x

class VertebraDataset(SuperbDataset):

    def __init__(self,
                 height: int,
                 width: int,
                 n_vertebrae: int,
                 vertebra_to_patient_idx: List[Dict[str, int]],
                 dataset: SuperbDataset,
                 patients_root: Path, 
                 removed: List[str] = [], 
                 patient_dirs: List[Path] = [], 
                 filter: Callable[[Any], bool] = lambda vertebra: True,
                 n_classes: int = 4, 
                 bbox_jitter: float = 0.1,
                 bbox_expansion: float = 0.1, 
                 bbox_format: str = 'cxcywh', 
                 bbox_normalization: bool = False,
                 random_expansion: bool = False,
                 ):
        
        super().__init__(patients_root=patients_root, 
                         removed=removed, 
                         patient_dirs=patient_dirs, 
                         filter=lambda patient: True, 
                         n_classes=n_classes, 
                         bbox_expansion=bbox_expansion, 
                         bbox_format=bbox_format, 
                         bbox_normalization=bbox_normalization, 
                         random_expansion=random_expansion, 
                         transforms=None,
                         imputer=None)
        
        self.height = height
        self.width = width
        self.n_vertebrae = n_vertebrae
        self.vertebra_to_patient_idx = vertebra_to_patient_idx
        self.superb = dataset

    @classmethod
    def from_fold(cls, 
                  height: int, 
                  width: int, 
                  config: Path, 
                  fold: int, 
                  split: str = "train", 
                  removed: List[str] = [], 
                  patient_dirs: List[Path] = [], 
                  filter: Callable[[Any], bool] = lambda vertebra: True,
                  n_classes: int = 4, 
                  bbox_expansion: float = 0.1, 
                  bbox_format: str = 'cxcywh', 
                  bbox_normalization: bool = False,
                  bbox_jitter: float = 0.1,
                  random_expansion: bool = False,
                  ) -> "VertebraDataset":
        

        # Get super classmethod
        dataset = SuperbDataset.from_fold(
            config=config, 
            fold=fold, 
            split=split, 
            removed=removed,
            filter=lambda patient: True,
            n_classes=n_classes,
            bbox_expansion=bbox_expansion,
            bbox_jitter=bbox_jitter,
            bbox_format=bbox_format,
            bbox_normalization=bbox_normalization,
            random_expansion=random_expansion,
            transforms=None,
            imputer=None
            )

        vertebra_to_patient_idx = []
        running_vertebra_idx = 0
        for patient_idx, (_, target) in enumerate(dataset):

            vertebra_idxs = target.indices.nonzero(as_tuple=False)

            for vertebra_idx in vertebra_idxs:

                visual_grade = target.visual_grades[vertebra_idx]

                # Map the running vertebra index to the patient index
                # and the internal vertebra index
                vertebra_data = {'patient_idx': patient_idx, 'vertebra_idx': vertebra_idx.item(), 'visual_grade': visual_grade.item()}

                if filter(vertebra_data):

                    vertebra_to_patient_idx.append(vertebra_data)

                    # Compute running vertebra index
                    running_vertebra_idx += 1

        n_vertebrae = running_vertebra_idx

        return cls(
                height=height,
                width=width,
                n_vertebrae=n_vertebrae,
                vertebra_to_patient_idx=vertebra_to_patient_idx,
                dataset=dataset,
                patients_root=dataset.patients_root,
                patient_dirs=patient_dirs,
                filter=filter,
                n_classes=n_classes,
                bbox_expansion=bbox_expansion,
                bbox_jitter=bbox_jitter,
                bbox_format=bbox_format,
                bbox_normalization=bbox_normalization,
                random_expansion=random_expansion,
            )
        
    def __len__(self) -> int:
        return self.n_vertebrae
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Target]:

        # Get the patient index and the vertebra index
        patient_idx     = self.vertebra_to_patient_idx[idx]['patient_idx']
        vertebra_idx    = self.vertebra_to_patient_idx[idx]['vertebra_idx']
        
        image, target = self.superb[patient_idx]
        
        # image = image.repeat(1, 3, 1, 1)

        # print(image)
        # image = normalize(image, SuperbStatistics.MAX).reshape(1, 1, *image.shape).repeat(1, 3, 1, 1)

        # Get the bounding boxes to crop image
        bounding_box    = target.boxes[vertebra_idx].unsqueeze(0) # (1, 4)
        keypoints       = target.keypoints.reshape(-1, 12)[vertebra_idx].unsqueeze(0) # (1, 12)
        label           = target.labels[vertebra_idx].unsqueeze(0) # (1, 1)
        name            = target.names[vertebra_idx] # (1, 1)
        visual_grade    = target.visual_grades[vertebra_idx].unsqueeze(0) # (1, 1)
        morphological_grade = target.morphological_grades[vertebra_idx].unsqueeze(0) # (1, 1)
        weight         = target.weights[vertebra_idx].unsqueeze(0) # (1, 1)

        # Align keypoints to bounding box
        keypoints = keypoints.reshape(bounding_box.shape[0], -1, 2) # (1, 6, 2)

        # Move keypoints to the origin
        x0, y0, x1, y1 = bounding_box[0]
        keypoints[:, :, 0] -= x0
        keypoints[:, :, 1] -= y0

        # Old-width to new-width ratio
        w_ratio = self.width / (x1 - x0)
        h_ratio = self.height / (y1 - y0)

        # Scale keypoints
        keypoints[:, :, 0] *= w_ratio
        keypoints[:, :, 1] *= h_ratio

        # Sort keypoints
        keypoints = keypoints.reshape(-1, 2)
        # keypoints = rotational_sort(keypoints).reshape(1, 6, 2)
        keypoints = as_ordered(keypoints).reshape(1, 6, 2)

        
        # Crop the image
        cropped = roi_align(image, [bounding_box], output_size=(self.height, self.width)).squeeze(0)

        return cropped, Target(
            boxes=bounding_box,
            keypoints=keypoints,
            labels=label,
            names=name,
            visual_grades=visual_grade,
            morphological_grades=morphological_grade,
            indices=target.indices,
            weights=weight,
            id=target.id,
        )

class VertebraDataModule(L.LightningDataModule):

    def __init__(self, 
                 height: int,
                 width: int,
                 source: Path,
                 batch_size: int,
                 removed: List[str],
                 bbox_expansion: float,
                 bbox_jitter: float,
                 bbox_normalization: bool,
                 bbox_format: Literal["cxcywh", "xyxy"],
                 n_classes: int,
                 fold: Optional[int] = None,
                 n_workers: int = 16,
                 ):
        
        super().__init__()

        self.height = height
        self.width = width
        self.fold = fold
        self.batch_size = batch_size
        self.source = source
        self.removed = removed
        self.bbox_expansion = bbox_expansion
        self.bbox_jitter = bbox_jitter
        self.bbox_normalization = bbox_normalization
        self.bbox_format = bbox_format
        self.n_classes = n_classes
        self.n_workers = n_workers

        
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        
        self.train_data = VertebraDataset.from_fold(self.height, self.width, 
                                            config=self.source,
                                            fold=self.fold,
                                            split="train",
                                            removed=self.removed,
                                            bbox_expansion=self.bbox_expansion,
                                            random_expansion=True,
                                            bbox_normalization=False,
                                            bbox_format="xyxy",
                                            bbox_jitter=self.bbox_jitter,
                                            n_classes=self.n_classes,
                                        )
        
        # Compute class frequencies
        if self.fold is None:
            print("Computing class frequencies...")
            grades = []
            types  = []
            for x, target in self.train_data:
                grades.append(target.visual_grades)
                types.append(target.labels)

            grades = torch.stack(grades).squeeze().tolist()
            types  = torch.stack(types).squeeze().tolist()

            grade_weights   = compute_class_weight(class_weight="balanced", classes=np.unique(grades), y=grades)
            type_weights    = compute_class_weight(class_weight="balanced", classes=np.unique(types), y=types)

            print(f"Grade weights:\t {grade_weights}")
            print(f"Type weights:\t{type_weights}")

    def setup(self, stage: Optional[str] = None) -> None:


        if stage == "fit" or stage == "val" or stage is None:
            # Load the dataset
            self.val_data = VertebraDataset.from_fold(self.height, self.width,
                                            config=self.source,
                                            fold=self.fold,
                                            split="val",
                                            removed=self.removed,
                                            bbox_expansion=self.bbox_expansion,
                                            random_expansion=False,
                                            bbox_normalization=False,
                                            bbox_format="xyxy",
                                            bbox_jitter=self.bbox_jitter,
                                            n_classes=self.n_classes,
                                        )
            
            print(f"Train:\t n_vertebra={len(self.train_data)},\t n_patients={len(self.train_data.superb)}")
            print(f"Val:\t n_vertebrae={len(self.val_data)},\t n_patients={len(self.val_data.superb)}")
                
        elif stage == "test":

            self.test_datasets = []
            for name, condition in [
                ("all", lambda x: x),  
                ("compressions", lambda x: x["visual_grade"] > 0), 
                ("only mild", lambda x: x["visual_grade"] == 1), 
                ("moderate and severe", lambda x: x["visual_grade"] > 1), 
                ("only moderate", lambda x: x["visual_grade"] == 2),
                ("only severe", lambda x: x["visual_grade"] == 3)]:

                dataset = VertebraDataset.from_fold(self.height, self.width,
                                            config=self.source,
                                            fold=self.fold,
                                            split="holdout",
                                            removed=self.removed,
                                            bbox_expansion=self.bbox_expansion,
                                            random_expansion=False,
                                            bbox_normalization=False,
                                            bbox_format="xyxy",
                                            bbox_jitter=self.bbox_jitter,
                                            n_classes=self.n_classes,
                                            filter=condition
                                        )
                
                self.test_datasets.append(dataset)
                
                print(f"Test ({name}):\t n_vertebrae={len(dataset)},\t n_patients={len(dataset.superb)}")
            

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, collate_fn=self.collate)
    
    def val_dataloader(self) -> DataLoader:
       return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, collate_fn=self.collate)
    
    def test_dataloader(self) -> DataLoader:
        return [DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_workers, pin_memory=True, collate_fn=self.collate) for dataset in self.test_datasets]
    
    def collate(self, batch: List[Tuple[Tensor, Target]]) -> Batch:
        x, y = zip(*batch)

        return Batch(
            x=torch.stack(x),
            y=y,
        )
    


