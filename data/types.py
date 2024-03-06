from pathlib import Path
import json
from typing import *

import numpy as np
from PIL import Image

from dataclasses import dataclass, asdict

from torchvision.transforms.functional import crop
from torchvision.ops import box_convert
import torch
from torch import Tensor, tensor

from .constants import VERTEBRA_NAMES

@dataclass
class Target:
    keypoints: Tensor
    boxes: Tensor
    names: List[str] | str
    visual_grades: Tensor
    morphological_grades: Tensor
    labels: Tensor
    types: Tensor
    indices: Tensor
    weights: Tensor
    id: Optional[str] = None

    def to(self, device: str) -> "Target":
        return Target(
            keypoints=self.keypoints.to(device),
            boxes=self.boxes.to(device),
            names=self.names,
            visual_grades=self.visual_grades.to(device),
            morphological_grades=self.morphological_grades.to(device),
            types=self.types.to(device),
            labels=self.labels.to(device),
            weights=self.weights.to(device),
            indices=self.indices.to(device),
            id=self.id,
        )
    
    def to_dict(self) -> Dict[str, Tensor]:
        return {
            "keypoints": self.keypoints,
            "boxes": self.boxes,
            "names": self.names,
            "visual_grades": self.visual_grades,
            "morphological_grades": self.morphological_grades,
            "types": self.types,
            "labels": self.labels,
            "weights": self.weights,
            "indices": self.indices,
            "id": self.id,
        }


@dataclass
class Batch:
    x: Tensor
    y: List[Target]
    original_sizes: List[Tuple[int, int]] = None

    def to(self, device: str) -> "Batch":
        return Batch(
            x=self.x.to(device),
            original_sizes=self.original_sizes,
            y=[_.to(device) for _ in self.y],
        )

@dataclass
class Prediction:
    mu: Tensor
    sigma: Optional[Tensor] = None

    def to(self, device: str) -> "Prediction":
        return Prediction(
            mu=self.mu.to(device),
            sigma=self.sigma.to(device) if self.sigma is not None else None,
        )
    
@dataclass
class Loss:
    keypoints: Tensor
    boxes: Optional[Tensor] = tensor(0.0)
    giou: Optional[Tensor] = tensor(0.0)
    cross_entropy: Optional[Tensor] = tensor(0.0)
    polynomial: Optional[Tensor] = tensor(0.0)

    def to(self, device: str) -> "Loss":
        return Loss(
            bboxes=self.boxes.to(device),
            giou=self.giou.to(device),
            keypoints=self.keypoints.to(device),
            cross_entropy=self.cross_entropy.to(device),
            polynomial=self.polynomial.to(device),
        )
    
    @property
    def total(self) -> Tensor:
        return self.boxes + self.keypoints + self.giou + self.cross_entropy + self.polynomial
    
    def to_dict(self) -> Dict[str, Tensor]:
        return {
            "keypoints": self.keypoints,
            "boxes": self.boxes,
            "giou": self.giou,
            "cross_entropy": self.cross_entropy,
            "polynomial": self.polynomial,
            "total": self.total
        }

@dataclass
class Output:
    keypoints: Optional[Prediction] = None
    bboxes: Optional[Prediction] = None
    logits: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


    def to(self, device: str) -> "Output":
        return Output(
            bboxes=self.bboxes.to(device),
            keypoints=self.keypoints.to(device),
            logits=self.logits.to(device),
            labels=self.labels.to(device),
            scores=self.scores.to(device),
        )
    
    def to_dict(self) -> Dict[str, Tensor]:
        return {
            "keypoints": self.keypoints,
            "boxes": self.bboxes,
            "logits": self.logits,
            "labels": self.labels,
            "scores": self.scores,
        }

@dataclass
class VertebraOutput:
    keypoints: Optional[Prediction] = None
    grade_logits: Optional[Tensor] = None
    type_logits: Optional[Tensor] = None


    def to(self, device: str) -> "Output":
        return Output(
            keypoints=self.keypoints.to(device),
            grade_logits=self.grade_logits.to(device),
            type_logits=self.type_logits.to(device),
        )
    
    def to_dict(self) -> Dict[str, Tensor]:
        return asdict(self)

@dataclass
class DXA:
    """
    A class representing a lateral/sagittal spine image.
    """

    path: Path

    @property
    def image(self) -> Image:
        return Image.open(self.path)
    
    @property
    def size(self) -> Tuple[int, int]:
        return self.image.size
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.image)
    
    def crop(self, bbox: "Bbox") -> np.ndarray:
        return crop(self.image, bbox.y, bbox.x, bbox.height, bbox.width)

    
@dataclass
class CT:
    """
    A class representing a CT image.
    """

    path: Path

    @property
    def image(self) -> Image:
        return Image.open(self.path)


@dataclass
class Point:
    """
    A class representing a point.
    """

    x: float
    y: float
    visible: int = 1

    def to_numpy(self, ignore_visibility: bool = False) -> np.ndarray:

        if ignore_visibility:
            return np.array([self.x, self.y])
        else:
            return np.array([self.x, self.y, self.visible])

    
    def __repr__(self) -> str:
        return f"Point(x={self.x:.2f}, y={self.y:.2f}, visible={self.visible})"

@dataclass
class Annotation:
    points: List[Point]
    image_height: Optional[float] = None
    image_width: Optional[float] = None

    def to_numpy(self, height: int = None, width: int = None) -> np.ndarray:

        normalize = height is not None and width is not None

        return np.array([
            (p.x / width, p.y / height) if normalize else (p.x, p.y) for p in self.points      
            ])
    
    def to_named(self, height: int = None, width: int = None) -> List[Tuple[float, float, str]]:

        points = []
        for ps, name in zip(
            [self.posterior, self.middle, self.anterior],
            ["posterior", "middle", "anterior"]
        ):
            for p in ps:
                
                points.append((
                    p.x / width if width is not None else p.x,
                    p.y / height if height is not None else p.y,
                    name))
            
        return points
    
    def to_list(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.points]
    
    def to_bbox(self) -> "Bbox":
        return Bbox.from_annotation(self)
    
    def __len__(self) -> int:
        return len(self.points) if self.points is not None else 0
    
    @property
    def posterior(self) -> List[Point]:
        return self.points[0:2]
    
    @property
    def middle(self) -> List[Point]:
        return self.points[2:4]
    
    @property
    def anterior(self) -> List[Point]:
        return self.points[4:6]
    
    @property
    def centroid(self) -> Point:
        return Point(np.mean([p.x for p in self.points]), np.mean([p.y for p in self.points]))
    
    def set_size(self, image_height: float, image_width: float) -> "Annotation":
        self.image_height = image_height
        self.image_width = image_width

        return self
    
    def resize(self, new_img_height: float, new_img_width: float):
        return Annotation([Point(p.x * new_img_width / self.image_width, p.y * new_img_height / self.image_height, p.visible) for p in self.points], new_img_height, new_img_width)
    

@dataclass
class Bbox:
    """
    A class representing a bounding box, with x, y, width and height. 
    Coordinates are in pixels, relative to the original image size.
    """

    x: float
    y: float
    width: float
    height: float
    image_height: Optional[float] = None
    image_width: Optional[float] = None

    @staticmethod
    def from_annotation(annotation: Annotation) -> "Bbox":
        points  = annotation.points

        x       = min([p.x for p in points])
        y       = min([p.y for p in points])
        width   = max([p.x for p in points]) - x
        height  = max([p.y for p in points]) - y

        return Bbox(x, y, width, height, None, None)
    
    def to_normalized(self, height: int, width: int) -> "NormalizedBbox":
        x = self.x / width
        y = self.y / height
        width = self.width / width
        height = self.height / height

        return NormalizedBbox(x, y, width, height, self.image_height, self.image_width)
    
    def to_expanded(self, expand: float = 0.2) -> "ExpandedBbox":
        x = self.x - expand * self.width
        y = self.y - expand * self.height
        width = self.width + 2 * expand * self.width
        height = self.height + 2 * expand * self.height

        return ExpandedBbox(x, y, width, height, self.image_height, self.image_width)
    
    def resize(self, new_img_height: float, new_img_width: float):
        
        x = self.x * new_img_width / self.image_width
        y = self.y * new_img_height / self.image_height
        width = self.width * new_img_width / self.image_width
        height = self.height * new_img_height / self.image_height

        return Bbox(x, y, width, height, new_img_height, new_img_width)
    
    def to_format(self, format: str) -> torch.Tensor:

        bbox = self.to_numpy()

        return box_convert(torch.Tensor(bbox), "xywh", format).numpy()
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.width, self.height], dtype=np.float32)
    

@dataclass
class NormalizedBbox(Bbox):
    """
    A class representing a bounding box, with x, y, width and height.
    Coordinates are normalized, relative to the original image size.
    """

    @staticmethod
    def from_annotation(annotation: Annotation, height: int, width: int) -> "NormalizedBbox":
        bbox = Bbox.from_annotation(annotation)

        x = bbox.x / width
        y = bbox.y / height
        width = bbox.width / width
        height = bbox.height / height

        return NormalizedBbox(x, y, width, height)
    
@dataclass
class ExpandedBbox(Bbox):
    """
    A class representing a bounding box, with x, y, width and height.
    Coordinates are in pixels, relative to the original image size.
    """
    @staticmethod
    def from_bbox(bbox: Bbox, expand: float = 0.1) -> "ExpandedBbox":
        x = bbox.x - expand * bbox.width
        y = bbox.y - expand * bbox.height
        width = bbox.width + 2 * expand * bbox.width
        height = bbox.height + 2 * expand * bbox.height

        return ExpandedBbox(x, y, width, height)
    
@dataclass
class Vertebra:
    name: str
    grad_morf: str
    grad_visuell: str
    typ: str
    ej_bedombbar: str
    kommentar: str
    coordinates: Optional[Annotation] = None

    
@dataclass
class Vertebrae:
    T4: Vertebra
    T5: Vertebra
    T6: Vertebra
    T7: Vertebra
    T8: Vertebra
    T9: Vertebra
    T10: Vertebra
    T11: Vertebra
    T12: Vertebra
    L1: Vertebra
    L2: Vertebra
    L3: Vertebra
    L4: Vertebra
    ULL: float
    K: float
    R: float
    A: float
    DF: float
    S: float
    BAC: float
    FR: float
    SK: float
    VLS: float
    HLS: float
    VBS: float
    HBS: float
    GRANSKAD: str
    Reader: str
    EJ_BEDÖMBAR: str
    pixel_spacing: Tuple[float, float]
    height: int
    width: int
    annotations: List[Annotation]

    def __iter__(self) -> Iterator[Vertebra]:
        for vertebra in VERTEBRA_NAMES:
            yield getattr(self, vertebra)

    @staticmethod
    def from_json(json_path: Path) -> "Vertebrae":
        with open(json_path) as f:
            data = json.load(f)

        vertebrae = {}
        for vertebra in VERTEBRA_NAMES:

            coords = data[vertebra].get("coordinates", None)
            if coords is not None:
                transformed = coords
                try:
                    annotation = Annotation([Point(*p) for p in transformed])
                except Exception as e:
                    annotation = Annotation([])
            else:
                annotation = Annotation([])

            vertebrae[vertebra] = Vertebra(
                vertebra,
                data[vertebra]["GRAD_MORF"],
                data[vertebra]["GRAD_VISUELL"],
                data[vertebra]["TYP"],
                data[vertebra]["_EJ_BEDÖMBAR"],
                data[vertebra]["KOMMENTAR"],
                annotation
            )

        annotations = []
        if data.get("keypoints") is not None:
            for v in data["keypoints"]:
                points = [Point(*p) for p in v]
                annotations.append(Annotation(points))

        other = {
            "ULL": data["ULL"],
            "K": data["K"],
            "R": data["R"],
            "A": data["A"],
            "DF": data["DF"],
            "S": data["S"],
            "BAC": data["BAC"],
            "FR": data["FR"],
            "SK": data["SK"],
            "VLS": data["VLS"],
            "HLS": data["HLS"],
            "VBS": data["VBS"],
            "HBS": data["HBS"],
            "GRANSKAD": data["GRANSKAD"],
            "Reader": data["Reader"],
            "EJ_BED\u00d6MBAR": data["EJ_BED\u00d6MBAR"],
            "pixel_spacing": tuple(data["pixel_spacing"]),
            "height": data.get("height", None),
            "width": data.get("width", None),
            "annotations": annotations
        }
        
        # print(other)

        
        return Vertebrae(**vertebrae, **other)
    
    def labels_to_numpy(self) -> np.ndarray:
        return np.array([v.grad_visuell for v in self.__dict__.values() if isinstance(v, Vertebra)])
    
    def coordinates_to_numpy(self) -> np.ndarray:
        return np.array([v.coordinates.to_numpy() for v in self.__dict__.values() if (isinstance(v, Vertebra) and len(v.coordinates) > 0)])


@dataclass
class Patient:
    moid: str
    root: Path
    spine: DXA
    leg: CT
    arm: CT
    vertebrae: Vertebrae

    @staticmethod
    def from_moid(moid: str, root: Path) -> "Patient":
        patient_dir = root / moid
        spine_path  = patient_dir / "lateral" / "patient.tiff"
        label_path  = patient_dir / "lateral" / "patient.json"
        dxa = DXA(spine_path)
        leg = None
        arm = None

        vertebrae= Vertebrae.from_json(label_path) if label_path.exists() else None
        
        return Patient(
            moid, 
            root, 
            dxa, 
            leg, 
            arm, 
            vertebrae
            )
        