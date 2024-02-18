
import warnings
import time
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning import Trainer

from data.vertebra import build_datamodule
from models.vertebra import build_model
from argparse import ArgumentParser
from models.vertebra.callbacks import VertebraPlotCallback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

def train(args):

    # Set seed
    seed_everything(args.seed)

    torch.set_float32_matmul_precision('medium')

    # Logging
    if args.log:
        logger = WandbLogger(
            name=args.name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S"),
            project=args.name,
            config=args,
            save_dir=args.log_dir,
        )
    else:
        logger = None

    # Callbacks for image logging and checkpointing
    callbacks = [
        ModelCheckpoint(
            monitor="val_stage/auroc",
            filename="{epoch:02d}-{step:02d}",
            save_top_k=2,
            mode="max",
        ),
        RichProgressBar(),
        RichModelSummary(),
        VertebraPlotCallback(
            n_samples=6, 
            plot_frequency=10, # Every k batches 
            )
    ]

    # Set up data module
    dm  = build_datamodule(args)

    # Set up and choose model
    args.class_weights = dm.class_weights
    model = build_model(args)

    # Set up trainer
    trainer = Trainer(
        accelerator="gpu" if args.device != "cpu" else args.device,
        devices = [args.device] if args.device != "cpu" else 1,
        max_epochs=args.n_epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run = args.debug if args.debug else False,
        # gradient_clip_val=args.grad_clip_norm,
        # profiler="advanced" if args.debug else None,
    )

    # Train
    if not args.test:
        trainer.fit(model, dm, ckpt_path=args.checkpoint if args.checkpoint else None)

    # Test
    trainer.test(model, dm, ckpt_path=args.checkpoint if args.checkpoint else None)


def main():

    parser = ArgumentParser()

    parser.add_argument("--source", type=str, default="./folds.csv")
    parser.add_argument("--cfg", type=str, default="./configs/data.json")
    parser.add_argument("--errors", type=str, default="./errors/reconstruction_errors.csv")

    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--target", type=str, default="keypoint")

    # Training parameters
    parser.add_argument("--model", type=str, default="vertebra-detr")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--train_fraction", type=int, default=0.85)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--bbox_expansion", type=float, default=0.4)
    parser.add_argument("--bbox_format", type=str, default="xyxy")
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_keypoints", type=int, default=6)
    parser.add_argument("--n_vertebrae", type=int, default=13)
    parser.add_argument("--n_dims", type=int, default=2)
    parser.add_argument("--missing_weight", type=float, default=1e-3)
    parser.add_argument("--p_augmentation", type=float, default=0.5)
    parser.add_argument("--fill_value", type=float, default=0)
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--prior", type=str, default="laplace")
    parser.add_argument("--rotation", type=float, default=45.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.8)
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--ce_image_loss_weight", type=float, default=1.0)
    parser.add_argument("--ce_loss_activation_epoch", type=int, default=-1)
    parser.add_argument("--class_weights", 
                        type=list,
                        default=[1.0, 1.0, 1.0, 1.0]
                         ) # List of class weights
    parser.add_argument("--test", action="store_true")

    # Training details
    parser.add_argument("--precision", type=int, default=64)
    parser.add_argument("--log_every_n_steps", type=int, default=5)
    
    # Logging
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--name", type=str, default="superb")
    parser.add_argument("--n_imgs_to_log", type=int, default=4)
    parser.add_argument("--log_imgs_every_n_epochs", type=int, default=1)

    parser.add_argument("--checkpoint", type=str, default="")


    # Detr arguments

    # parser.add_argument('--backbone_name', default='swin_tiny', type=str,
    #                     help="Name of the deit backbone to use")    
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_keypoint', default=5, type=float,
                        help="keypoint coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--keypoint_loss_coef', default=5, type=float)
    parser.add_argument('--polynomial_loss_coef', default=1000, type=float)
    
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='superb')
    # parser.add_argument('--coco_path', type=str)
    # parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--device', default='cuda',
                        # help='device to use for training / testing')
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Get arguments
    args = parser.parse_args()

    # Train
    train(args)


if __name__ == "__main__":

    main()