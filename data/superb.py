import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from pathlib import Path
from enum import Enum
import json

from typing import *
from .types import *

import numpy as np
import pandas as pd

from utils import bbox_from_keypoints
from .augmentation import build_augmenter

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torchvision.ops import box_convert

class Stage(str, Enum):

    FIT     = 'fit'
    TRAIN   = 'train'
    VAL     = 'validate'
    TEST    = 'test'


class PatientDataset(Dataset):

    def __init__(self, patients_root: Path, removed: List[str] = [], patient_dirs: List[Path] = []) -> None:
        super().__init__()

        self.patients_root = patients_root
        self.patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in removed)] \
                    if not patient_dirs else patient_dirs
        
        self.patients = [Patient.from_moid(patient_dir.name, patients_root) for patient_dir in self.patient_dirs]

    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def __getitem__(self, index) -> Patient:
        return self.patients[index]


class SuperbData(Dataset):

    def __init__(self, 
                 patients_root: Path,
                 removed: List[str] = [],
                 size: Tuple[float, float] = (1700, 700),
                 patient_dirs: List[Path] = [],
                 bbox_expansion: float = 0.1,
                 bbox_format: str = 'cxcywh',
                 transforms = None,
                 bbox_normalization: bool = False,
                 n_classes: int = 4,
                 imputer = None
                 ) -> None:
        super().__init__()

        self.patients_root = patients_root # Root directory of all patient directories
        self.removed = removed # List of patient directories to remove

        # Get all patient directories, unless specified
        self.patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in self.removed)] \
                    if not patient_dirs else patient_dirs

        self.size = size # Resize images to this size
        self.new_height, self.new_width = size 
        self.n_classes = n_classes # Number of classes
        self.bbox_expansion = bbox_expansion # Expand bounding boxes by this factor to ensure vertebra is included
        self.bbox_format = bbox_format # Format of bounding boxes, either 'xyxy' or 'xywh'
        self.bbox_normalization = bbox_normalization # Normalize bounding boxes to [0, 1]
        self.transforms = transforms # Augmentation transforms
        self.imputer: KNNImputer = imputer # Imputer for missing keypoints


    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def _observables_from_patient(self, patient: Patient) -> Target:
        """
        Extracts the observables from a patient, used for detection.
        
        Args:
            patient (Patient): Patient to extract observables from.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of observables, with keys:
                - Target.KEYPOINT.value: Tensor of keypoints with shape (N, 6, 2)
                - Target.BBOX.value: Tensor of bounding boxes with shape (N, 4)
                - Target.LABEL.value: Tensor of labels with shape (N,)
                - Target.INDEX.value: Tensor of indices with shape (N,)
        """

        keypoints = []
        bboxes = []
        labels = []
        visual_grade = []
        morphological_grade = []
        names = []
        indices = torch.zeros((len(VERTEBRA_NAMES)), dtype=torch.bool)
        for i, v in enumerate(patient.vertebrae):

            # If the vertebra is annotated
            if len(v.coordinates) > 0:

                keypoints.append(v.coordinates.to_numpy())
                bboxes.append(v.coordinates.to_bbox().to_format("xyxy"))
                labels.append(int(v.typ) if v.typ is not None else 0)
                visual_grade.append(int(v.grad_visuell) if v.grad_visuell is not None else -1)
                morphological_grade.append(int(v.grad_morf) if v.grad_morf is not None else -1)
                names.append(v.name)
                indices[VERTEBRA_NAMES.index(v.name)] = True

            else:
                keypoints.append(np.full((6, 2), np.nan))
                bboxes.append(np.full((4), np.nan))
                labels.append(0)
                visual_grade.append(-1)
                morphological_grade.append(-1)
                names.append(v.name)
                indices[VERTEBRA_NAMES.index(v.name)] = False

        keypoints = torch.tensor(np.array(keypoints), dtype=torch.float32)
        bboxes = torch.tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        visual_grade = torch.tensor(visual_grade, dtype=torch.long)
        morphological_grade = torch.tensor(morphological_grade, dtype=torch.long)

        target = Target(
            keypoints=keypoints,
            boxes=bboxes,
            labels=labels,
            visual_grades=visual_grade,
            morphological_grades=morphological_grade,
            names=names,
            indices=indices,
            id=patient.moid
        )

        return target


    def __getitem__(self, index) -> Tuple[Tensor, Target]:
                
            p = Patient.from_moid(self.patient_dirs[index].name, self.patients_root)
            image = np.asarray(p.spine.image)

            # Get observables
            target = self._observables_from_patient(p)

            # Impute missing keypoints
            if self.imputer is not None:
                keypoints = target.keypoints.reshape(1, -1).numpy()
                keypoints = torch.from_numpy(self.imputer.transform(keypoints)).reshape(-1, 6, 2)
                target.keypoints = keypoints
                
                # Compute bounding boxes from keypoints (xyxy)
                bboxes = bbox_from_keypoints(keypoints.numpy())
                target.boxes = torch.from_numpy(bboxes)

            
            # Apply augmentation transforms
            if self.transforms is not None:
                image, keypoints, bboxes = self.transforms(image, target.keypoints, target.boxes)
            else:
                keypoints, bboxes = target.keypoints, target.boxes

            # Normalize bounding boxes to [0, 1]
            if self.bbox_normalization:

                bboxes = bboxes / torch.tensor([self.new_width, self.new_height, self.new_width, self.new_height])
                keypoints = keypoints.view(-1, 2) / torch.tensor([self.new_width, self.new_height])
                
            # Convert to new format
            if self.bbox_format != 'xyxy':
                bboxes = box_convert(bboxes, "xyxy", self.bbox_format)

            try:
                keypoints   = keypoints.view(-1, 6*2) # (N, 6*2)
            except:
                raise ValueError(f"Keypoints shape: {keypoints.shape}")
            
            target = Target(
                keypoints=keypoints,
                boxes=bboxes,
                labels=target.labels,
                visual_grades=target.visual_grades,
                morphological_grades=target.morphological_grades,
                names=target.names,
                indices=target.indices,
                id=target.id
            )

            return image, target

            # return image, bboxes, keypoints, labels, indices, moid
    
    def get_patient(self, moid: str) -> Patient:
        return Patient.from_moid(moid, self.patients_root)
    
    def where_label(self, condition: Callable[[Patient], bool]) -> List[Patient]:
        
        patients = []
        for i, patient_dir in enumerate(self.patient_dirs):
            patient = self.get_patient(patient_dir.name)
            if condition(patient):
                patients.append(patient)

        return patients
    
    def filter(self, condition: Callable[[Patient], bool]) -> "SuperbData":
        
        patients = self.where_label(condition)
        patient_dirs = [patient.root / patient.moid for patient in patients]

        ds = SuperbData(
            self.patients_root,
            self.removed,
            self.size,
            patient_dirs,
            n_classes=self.n_classes,
            bbox_expansion=self.bbox_expansion,
            bbox_format=self.bbox_format,
            bbox_normalization=self.bbox_normalization,
            transforms=self.transforms,
            imputer=self.imputer
        )

        return ds
    


class SuperbDataModule(L.LightningDataModule):

    def __init__(self, 
                 data_dir: Path,
                 batch_size: int = 1,
                 image_size: Tuple[int, int] = (600, 280),
                 train_split: float = 0.8,
                 removed: List[str] = [],
                 bbox_expansion: float = 0.1,
                 bbox_format: str = 'cxcywh',
                 bbox_normalization: bool = False,
                 n_classes: int = 4,
                 n_keypoints: int = 6,
                 n_workers: int = 8,
                 transforms: List = [None, None],
                 n_neighbors: int = 10,
                 filter: Callable[[Patient], bool] = lambda patient: True
                 ) -> None:
        
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_split = train_split
        self.bbox_expansion = bbox_expansion
        self.bbox_format = bbox_format
        self.bbox_normalization = bbox_normalization
        self.n_workers = n_workers
        self.removed = removed
        self.n_classes = n_classes
        self.n_keypoints = n_keypoints

        self.train_idxs = []
        self.val_idxs = []

        self.save_hyperparameters(
            "batch_size",
            "image_size",
            "train_split",
            "bbox_expansion",
            "bbox_format",
            "n_workers",
            "removed",
            "n_classes",
            "n_keypoints"
        )

        self.train_transforms, self.val_transforms = transforms

        self.filter = filter
        self.n_neighbors = n_neighbors

    def setup(self, stage: Stage) -> None:

        # Create a provistionary dataset
        data = SuperbData(
            self.data_dir, 
            size=self.image_size, 
            removed=self.removed,
            bbox_expansion=self.bbox_expansion,
            bbox_format=self.bbox_format,
            bbox_normalization=False,
            transforms=None,
            n_classes=self.n_classes,
            )

        data = data.filter(self.filter)

        self.train_idxs, self.val_idxs = train_test_split(
                np.arange(len(data)), 
                test_size=1 - self.train_split
                )
        
        # Train dataset to used for training as well as imputation
        filtered = data.filter(lambda p: p)
        keypoints = torch.cat([t.keypoints for i, (x, t) in enumerate(filtered) if i in self.train_idxs], dim=0).to("cpu").reshape(len(self.train_idxs), -1).numpy()

        # Fit imputer
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(keypoints)

        if stage == Stage.FIT:
            data = SuperbData(
                self.data_dir, 
                size=self.image_size, 
                removed=self.removed,
                bbox_expansion=self.bbox_expansion,
                bbox_format=self.bbox_format,
                bbox_normalization=self.bbox_normalization,
                transforms=None,
                n_classes=self.n_classes,
                imputer=self.imputer
                )
            
            self.train              = data.filter(lambda p: p)
            self.train.transforms   = self.train_transforms
            self.train.imputer      = self.imputer
            self.val                = data.filter(lambda p: p)
            self.val.imputer        = self.imputer
            self.val.transforms     = self.val_transforms

        if stage == Stage.VAL:

            self.val = data.filter(lambda p: p)
            self.val.transforms = self.val_transforms


    def train_dataloader(self) -> DataLoader:

        sampler = SubsetRandomSampler(self.train_idxs)

        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            sampler=sampler,
            collate_fn=self.collate,
            pin_memory=False
            )
    
    def val_dataloader(self) -> DataLoader:

        sampler = SubsetRandomSampler(self.val_idxs)

        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            sampler=sampler,
            collate_fn=self.collate,
            pin_memory=False
            )
    
    def collate(self, batch: List[Tuple[Tensor, Target]]) -> Batch:

        
        images = torch.stack([b[0] for b in batch]).squeeze(1)

        targets = [b[1] for b in batch]

        # Impute missing keypoints
        # keypoints = torch.cat([t.keypoints for t in targets], dim=0).to("cpu").reshape(len(targets), -1).numpy()
        # keypoints = torch.from_numpy(self.imputer.transform(keypoints)).to(images.device).reshape(len(targets), -1, 2)

        # Update targets
        # for i, target in enumerate(targets):
            # target.keypoints = keypoints[i]

        return Batch(
            x=images,
            y=targets
        )

def build_datamodule(args) -> SuperbDataModule:
    
    train_augmenter, val_augmenter = build_augmenter(
        p=args.p_augmentation,
        height=args.height,
        width=args.width,
        max_val=SuperbStatistics.MAX,
        fill_value=args.fill_value

    )

    errors = pd.read_csv(args.errors)
    error_moid   = errors.moid.values

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    # Remove patients with errors

    removed = [
        *cfg["removed"],
         *error_moid]

    # Set up data module
    dm = SuperbDataModule(
        Path(args.source),
        image_size=(args.height, args.width),
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        removed=removed,
        bbox_expansion=args.bbox_expansion,
        bbox_normalization=True,
        bbox_format=args.bbox_format,
        n_classes=args.n_classes,
        transforms=[train_augmenter, val_augmenter],
        filter= lambda p: not all([len(v.coordinates)==0 for v in p.vertebrae]) \
                         and all(len(v.coordinates) % 6 == 0 for v in p.vertebrae))
        # filter= lambda p: not any([len(v.coordinates)==0 for v in p.vertebrae]))

    dm.setup(stage="fit")

    return dm

