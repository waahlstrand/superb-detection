from .models import VertebraeDetector, VertebraViDT
from .criterion import RLELoss

def build_model(args):

    if args.model == "vertebra-detector":

        model = VertebraeDetector(
            n_vertebrae=13,
            n_keypoints=6,
            n_dims=2,
            backbone=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            criterion=RLELoss,
            lr=args.lr,
            lr_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
        )

    elif args.model == "vertebra-vidt":

        model = VertebraViDT(
                 args, 
                 n_classes= args.n_classes, 
                 n_keypoints = 6, 
                 n_dim = 2, 
                 n_channels = 3,
        )

    else:
        raise NotImplementedError
    
    return model