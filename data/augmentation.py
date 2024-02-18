#%%
import torch
import torch.nn as nn
from torch import Tensor
from typing import *
import numpy as np
import kornia.augmentation as K
from kornia.geometry.keypoints import Keypoints
from kornia.geometry.boxes import Boxes


class Normalize:

    def __init__(self):
        super().__init__()

    def __call__(self, input: torch.Tensor) -> torch.Tensor:

        x = torch.clone(input)

        max_value = torch.amax(x, axis=(2, 3), keepdims=True)
        min_value = torch.amin(x, axis=(2, 3), keepdims=True)
        x = (x - min_value) / (max_value - min_value)

        return x
    
class CleverCropAndResize:

    def __init__(self, n_manual = None) -> None:
        
        super().__init__()

        self.n_manual = n_manual
        

    def crop_around_bboxes(self, image: torch.Tensor, targets: Dict[str, torch.Tensor], n_manual: int = None):

        height, width = image.shape[1:]

        n_targets = len(targets["boxes"]) 

        # Select random number of consecutive vertebrae
        n = torch.randint(1, n_targets + 1, (1,)).item() if n_manual is None else n_manual

        # Select start index
        start = torch.randint(0, n_targets - n + 1, (1,)).item()

        # Select end index
        end = start + n

        # Select the targets
        bboxes = targets["boxes"][start:end]

        image_height, image_width = image.shape[1:]

        # Get the min and max x and y coordinates
        x_min = bboxes[:, 0].min().int().item()
        x_max = bboxes[:, 2].max().int().item()
        y_min = bboxes[:, 1].min().int().item()
        y_max = bboxes[:, 3].max().int().item()

        # Random expansion within the image
        max_expansion = 0.2

        # Keeping aspect ratio, expand the bounding box
        expansion = torch.rand(1).item() * max_expansion
        x_min = max(0, x_min - int(expansion * (x_max - x_min)))
        x_max = min(image_width, x_max + int(expansion * (x_max - x_min)))
        y_min = max(0, y_min - int(expansion * (y_max - y_min)))
        y_max = min(image_height, y_max + int(expansion * (y_max - y_min)))

        # Crop the image
        cropped_image = image[:, y_min:y_max, x_min:x_max]

        # Resize the image to original size
        old_height, old_width = cropped_image.shape[1:]
        cropped_image = torch.nn.functional.interpolate(cropped_image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

        # Crop the targets and resize them
        cropped_targets = {
            "boxes": targets["boxes"][start:end] - torch.tensor([x_min, y_min, x_min, y_min], device=targets["boxes"].device),
            "keypoints": targets["keypoints"][start:end] - torch.tensor([x_min, y_min, 0], device=targets["keypoints"].device),
            "labels": targets["labels"][start:end],
        }

        cropped_targets["boxes"][:, 0] *= width / old_width
        cropped_targets["boxes"][:, 1] *= height / old_height
        cropped_targets["boxes"][:, 2] *= width / old_width
        cropped_targets["boxes"][:, 3] *= height / old_height

        cropped_targets["keypoints"][:, :, 0] *= width / old_width
        cropped_targets["keypoints"][:, :, 1] *= height / old_height
  
        return cropped_image, cropped_targets

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        
        cropped_images = []
        cropped_targets = []
        for image, target in zip(images, targets):
            cropped_image, cropped_target = self.crop_around_bboxes(image, target, n_manual=self.n_manual)
            cropped_images.append(cropped_image)
            cropped_targets.append(cropped_target)

        return torch.stack(cropped_images), cropped_targets
    

class Augmenter(nn.Module):

    def __init__(self, 
                 p: float,
                 scale: float,
                 max_val: float
                 ) -> "Augmenter":
        
        super().__init__()

        self.p          = p
        self.scale      = scale
        self.max_val    = max_val

    @torch.no_grad()
    def forward(self, image: np.ndarray, keypoints: Tensor, bboxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Performs augmentation on the images, keypoints and bounding boxes, using Kornia.

        Args: 
            images: Tensor of shape (1, height, width)
            keypoints: Tensor of shape (n_keypoints, 2)
            bboxes: Tensor of shape (n_bboxes, 4)

        Returns:
            images: Tensor of shape (height, width)
            keypoints: Tensor of shape (n_keypoints, 2)
            bboxes: Tensor of shape n_bboxes, 4)
        """

        
        # image_max   = image.max().astype(np.float32)
        # max_val     = torch.max(torch.tensor(self.max_val), torch.tensor(image_max)).numpy()

        # image = torch.from_numpy(image.copy() / max_val).type(keypoints.dtype)

        # Kornia expects (H, W) but we have (W, H), we need to transpose
        # image = image

        image_height, image_width = image.shape[-2:]

        # Rescale
        new_width   = int(image_width * self.scale)
        new_height  = int(image_height * self.scale)

        # Random height crop
        height_crop_factor = torch.FloatTensor(1).uniform_(0.5, 1.0).item()


        model = K.AugmentationSequential(
            K.Resize((new_height, new_width), antialias=True),
            K.RandomInvert(p=self.p),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=self.p),
            K.RandomHorizontalFlip(p=self.p),
            K.RandomVerticalFlip(p=self.p),
            K.RandomRotation(degrees=5.0, p=self.p),
            K.RandomEqualize(p=self.p),
            K.RandomSharpness(p=self.p),
            # K.RandomCrop((int(new_height * height_crop_factor), new_width), p=self.p),
            # K.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=self.p),
            data_keys=["image", "keypoints", "bbox_xyxy"],
        )

        # Transform to Kornia format
        keypoints   = Keypoints(keypoints)
        bboxes      = Boxes.from_tensor(bboxes.unsqueeze(0), "xyxy")

        # Augment
        image, keypoints, bboxes = model(image, keypoints, bboxes)

        # Transform back to PyTorch format
        keypoints = keypoints.data
        bboxes = bboxes.to_tensor("xyxy")
        image = image

        return image, keypoints, bboxes

def build_augmenter(
                 p: float,
                 scale: float,
                 max_val: float
                ) -> Tuple[nn.Module, nn.Module]:
    
    return Augmenter(p, scale, max_val), Augmenter(0, scale, max_val)
    

    



