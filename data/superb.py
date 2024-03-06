import lightning as L
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from enum import Enum
import json
from rich import print

from typing import *

from .types import Patient
from .types import *
from .constants import SuperbStatistics
import numpy as np
import pandas as pd

from utils import patient_to_target, apply_imputation, normalize_bbox, normalize_keypoints
from models.backbones.DINO.util.misc import nested_tensor_from_tensor_list
from .augmentation import build_augmenter

from sklearn.impute import KNNImputer
from torchvision.ops import box_convert

def normalize(x: np.ndarray, max_val: float) -> Tensor:

    x_max   = x.max().astype(np.float32)
    max_val = torch.max(torch.tensor(max_val), torch.tensor(x_max)).numpy()

    x = torch.from_numpy(x.copy() / max_val).type(torch.float32)

    return x

class Stage(str, Enum):

    FIT     = 'fit'
    TRAIN   = 'train'
    VAL     = 'validate'
    TEST    = 'test'

class PatientDataset(Dataset):

    def __init__(self, patients_root: Path, removed: List[str] = [], patient_dirs: List[Path] = [],  filter: Callable[[Patient], bool] = lambda x: True) -> None:
        super().__init__()

        self.patients_root = patients_root
        patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in removed)] \
                    if not patient_dirs else patient_dirs
        
        self.patients = []
        self.patient_dirs = []
        for patient_dir in patient_dirs:
            patient = Patient.from_moid(patient_dir.name, patients_root)
            if filter(patient):
                self.patients.append(patient)
                self.patient_dirs.append(patient_dir)

    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def __getitem__(self, index) -> Patient:
        return self.patients[index]

class SuperbDataset(PatientDataset):

    def __init__(self, 
                 patients_root: Path, 
                 removed: List[str] = [], 
                 patient_dirs: List[Path] = [], 
                 filter: Callable[[Patient], bool] = lambda patient: True,
                 n_classes: int = 4, 
                 bbox_expansion: float = 0.1, 
                 bbox_format: str = 'cxcywh', 
                 bbox_normalization: bool = False,
                 bbox_jitter: float = 0.1,
                 random_expansion: bool = False,
                 transforms: nn.Module = None,
                 imputer: KNNImputer = None,
                 missing_weight: float = 1e-4) -> None:
        
        super().__init__(patients_root, removed, patient_dirs, filter=filter)

        self.n_classes = n_classes
        self.bbox_expansion = bbox_expansion
        self.bbox_jitter = bbox_jitter
        self.bbox_format = bbox_format
        self.bbox_normalization = bbox_normalization
        self.random_expansion = random_expansion
        self.missing_weight = missing_weight

        # Define an imputation function
        self.imputer = imputer # Imputer for missing keypoints

        # Define augmentation transforms
        self.transforms = transforms


    @classmethod
    def from_fold(cls, config: Path, fold: int, split: str = "train", **kwargs) -> "SuperbDataset":

        # Load the config csv
        config = pd.read_csv(config)

        if split == "train":
            config = config[config.train == 1]

        elif split == "val":
            config = config[config.val == 1]

        elif split == "holdout":
            config = config[config.holdout == 1]
            fold   = -1

        else:
            raise ValueError(f"Split {split} not recognized")
        
        # Filter by fold and split
        if fold is not None:
            config = config[config.fold == fold]

        # Get the patient directories
        patient_dirs    = [Path(p) for p in config.dir.values]

        patients_root   = patient_dirs[0].parent

        return cls(patients_root, patient_dirs=patient_dirs, **kwargs)
    
    def __getitem__(self, idx) -> Tuple[Tensor, Target]:
        
        patient = super().__getitem__(idx)

        image   = np.asarray(patient.spine.image)
        target  = patient_to_target(patient, 
                                    self.bbox_expansion,
                                    self.bbox_jitter,
                                    self.random_expansion,
                                    self.missing_weight)
        
        # Modify labels to accommadate for the different number of classes
        if self.n_classes == 3:
            target.labels[target.labels == 3] = 1
            target.types[target.types == 3] = 1


        elif self.n_classes == 4:
            target.labels = target.visual_grades # hack

        elif self.n_classes == 1:
            target.labels = torch.zeros_like(target.labels)
            target.types[target.types == 3] = 1
        else:
            raise ValueError("Number of classes must be either 1 or 3")
        
        image = normalize(image, SuperbStatistics.MAX).reshape(1, 1, *image.shape).repeat(1, 3, 1, 1)
                
        # Impute keypoints if imputer is defined
        if self.imputer is not None:
            
            nan_idxs = torch.isnan(target.keypoints).any(dim=-1).any(dim=-1)
            
            keypoints, boxes = apply_imputation(target.keypoints, self.imputer)

            # Update only the keypoints that were imputed
            target.keypoints[nan_idxs] = keypoints[nan_idxs]
            target.boxes[nan_idxs] = boxes[nan_idxs]


        # Apply transforms
        if self.transforms is not None:
            image, target.keypoints, target.boxes = self.transforms(image, target.keypoints, target.boxes)
        
        # Normalize bounding boxes to [0, 1]
        if self.bbox_normalization:
            height, width       = image.shape[-2], image.shape[-1]
            target.keypoints    = normalize_keypoints(target.keypoints, height, width)
            target.boxes        = normalize_bbox(target.boxes, height, width)

        # Convert to new format
        if self.bbox_format != 'xyxy':
            target.boxes = box_convert(target.boxes, "xyxy", self.bbox_format)
        
        return image, target

class SuperbDataModule(L.LightningDataModule):

    def __init__(self, 
                 source: Path,
                 fold: int = 0,
                 batch_size: int = 1,
                 train_split: float = 0.8,
                 removed: List[str] = [],
                 bbox_expansion: float = 0.1,
                 bbox_format: str = 'cxcywh',
                 bbox_normalization: bool = False,
                 p_augmentation: float = 0.5,
                 scale: float = 0.5,
                 n_classes: int = 4,
                 n_keypoints: int = 6,
                 n_workers: int = 8,
                 n_neighbors: int = 10,
                 missing_weight: float = 1e-4,
                 filter: str = "not_any"
                 ) -> None:
        
        super().__init__()

        self.source         = source
        self.fold           = fold
        self.batch_size     = batch_size
        self.train_split    = train_split
        self.bbox_expansion = bbox_expansion
        self.bbox_format    = bbox_format
        self.bbox_normalization = bbox_normalization
        self.n_workers      = n_workers
        self.removed        = removed
        self.n_classes      = n_classes
        self.n_keypoints    = n_keypoints
        self.missing_weight = missing_weight

        if filter == "not_any":
            filter = lambda p: not any([len(v.coordinates)==0 for v in p.vertebrae])
        elif filter == "not_all":
            filter = lambda p: not all([len(v.coordinates)==0 for v in p.vertebrae])
        else:
            raise NotImplementedError(f"Filter {filter} not implemented")


        self.train_idxs = []
        self.val_idxs = []

        self.save_hyperparameters(
            "batch_size",
            "train_split",
            "bbox_expansion",
            "bbox_format",
            "n_workers",
            "removed",
            "n_classes",
            "n_keypoints"
        )

        self.train_transforms, self.val_transforms = build_augmenter(
        p=p_augmentation,
        scale=scale,
        max_val=SuperbStatistics.MAX,
    )

        self.filter = filter
        self.n_neighbors = n_neighbors

        # Compute class weights
        self.data = SuperbDataset.from_fold(
            self.source, 
            fold=self.fold, 
            split="train", 
            bbox_expansion=self.bbox_expansion, 
            bbox_format=self.bbox_format,
            bbox_normalization=self.bbox_normalization,
            random_expansion=False,
            n_classes=self.n_classes,
            transforms=None,
            missing_weight=self.missing_weight,
            filter=self.filter
        )
        
        # Train dataset to used for training as well as imputation
        labels    = torch.cat([target.labels for image, target in self.data if len(target.keypoints) > 0], dim=0).to("cpu")       # Compute class frequency from training set
        
        vals, counts = torch.unique(labels, return_counts=True)
        self.class_weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)

        # Pretty print the vertebrae classes and their frequencies
        print("Vertebrae class frequencies:")
        for i, (v, c, w) in enumerate(zip(vals, counts, self.class_weights)):
            print(f"Class: {v.item()}\t Counts: {c.item()}\t Weight: {w.item()}")
        
    def setup(self, stage: Stage) -> None:

        keypoints = torch.cat([target.keypoints for image, target in self.data], dim=0).to("cpu")
        keypoints = keypoints.reshape(-1, 12).numpy()
        
        # Fit imputer
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(keypoints)

        if stage == Stage.FIT:

            self.train_data = SuperbDataset.from_fold(
                self.source, 
                fold=self.fold, 
                split="train", 
                bbox_expansion=self.bbox_expansion,
                random_expansion=False, 
                bbox_format=self.bbox_format,
                bbox_normalization=self.bbox_normalization,
                n_classes=self.n_classes,
                transforms=self.train_transforms,
                filter=self.filter,
                imputer=self.imputer,
                missing_weight=self.missing_weight
            )

            self.val_data = SuperbDataset.from_fold(
                self.source, 
                fold=self.fold, 
                split="val", 
                bbox_expansion=self.bbox_expansion, 
                random_expansion=False,
                bbox_format=self.bbox_format,
                bbox_normalization=self.bbox_normalization,
                n_classes=self.n_classes,
                transforms=self.val_transforms,
                filter=self.filter,
                imputer=self.imputer,
                missing_weight=self.missing_weight
            )

            print(f"Fold: {self.fold}")
            print(f"Train size: {len(self.train_data)}")
            print(f"Val size: {len(self.val_data)}")

        elif stage == Stage.TEST:

            self.test_data = SuperbDataset.from_fold(
                self.source, 
                fold=self.fold, 
                split="holdout", 
                bbox_expansion=self.bbox_expansion, 
                random_expansion=False,
                bbox_format=self.bbox_format,
                bbox_normalization=self.bbox_normalization,
                n_classes=self.n_classes,
                transforms=self.val_transforms,
                filter=self.filter,
                imputer=self.imputer,
                missing_weight=self.missing_weight
            )
            
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            collate_fn=self.collate,
            pin_memory=True
            )
    
    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            collate_fn=self.collate,
            pin_memory=True
            )
    
    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            collate_fn=self.collate,
            pin_memory=True
            )
    
    def collate(self, batch: List[Tuple[Tensor, Target]]) -> Batch:

        
        images = nested_tensor_from_tensor_list([b[0].squeeze() for b in batch])
        original_sizes = [b[0].shape[-2:] for b in batch]
        targets = [b[1] for b in batch]

        # # Impute missing keypoints
        # keypoints = torch.cat([t.keypoints for t in targets], dim=0).to("cpu").reshape(len(targets), -1).numpy()
        # keypoints = torch.from_numpy(self.imputer.transform(keypoints)).to(images.tensors.device).reshape(len(targets), -1, 2)

        # # Compute new bounding boxes from keypoints
        # bboxes = torch.cat([bbox_from_keypoints(k) for k in keypoints.reshape(len(targets), -1, 6, 2)], axis=0)
        # bboxes = bboxes.reshape(len(targets), -1, 4)

        # # Update targets
        # for i, target in enumerate(targets):
        #     target.keypoints = keypoints[i]
        #     target.boxes = bboxes[i]

        return Batch(
            x=images,
            y=targets,
            original_sizes=original_sizes
        )

def build_datamodule(args) -> SuperbDataModule:
    
    train_augmenter, val_augmenter = build_augmenter(
        p=args.p_augmentation,
        scale=args.scale,
        max_val=SuperbStatistics.MAX,
    )

    errors = pd.read_csv(args.errors)
    error_moid   = errors.moid.values

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    # Remove patients with errors
    removed = [
        *cfg["removed"],
         *error_moid]
    
    # Filter function
    if args.filter == "not_any":
        filter = lambda p: not any([len(v.coordinates)==0 for v in p.vertebrae])
    elif args.filter == "not_all":
        filter = lambda p: not all([len(v.coordinates)==0 for v in p.vertebrae])
    else:
        raise NotImplementedError(f"Filter {args.filter} not implemented")

    # Set up data module
    dm = SuperbDataModule(
        Path(args.source),
        fold=args.fold,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        removed=removed,
        bbox_expansion=args.bbox_expansion,
        bbox_normalization=True,
        bbox_format=args.bbox_format,
        n_classes=args.n_classes,
        transforms=[train_augmenter, val_augmenter],
        filter=filter,
        missing_weight=args.missing_weight
        )

    return dm

