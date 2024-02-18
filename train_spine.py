
import warnings
import time
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, EarlyStopping
from lightning import Trainer

from data.superb import build_datamodule
from models.spine import build_model
from argparse import ArgumentParser
from models.callbacks import SpinePlotCallback
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
            project="superb_spine",
            config=args,
            save_dir=args.log_dir,
        )
    else:
        logger = None

    # Callbacks for image logging and checkpointing
    callbacks = [
        ModelCheckpoint(
            monitor="val_stage/loss",
            filename="{epoch:02d}-{step:02d}",
            save_top_k=2,
            mode="min",
        ),
        # EarlyStopping(
        #     monitor="val_stage/loss_giou",
        #     patience=7,
        #     mode="min",
        #     verbose=True,
        # ),
        RichProgressBar(),
        RichModelSummary(),
        SpinePlotCallback(
            n_samples=4, 
            plot_frequency=10, # Every k batches 
            save_to_disk=args.debug,
            )
    ]

    # Set up data module
    dm_all  = build_datamodule(args)

    # args.filter = "not_any"
    # dm_any  = build_datamodule(args)

    # Set up and choose model
    model = build_model(args, class_weights=dm_all.class_weights)


    # Set up trainer
    if args.device == "cpu":
        accelerator = "cpu"
        device = 1
    elif args.device == "gpu" or args.device == "cuda":
        accelerator = "gpu"
        device = 1
    else:
        raise ValueError("Device must be either 'cpu', 'gpu' or 'cuda'")

    trainer = Trainer(
        accelerator=accelerator,
        # accelerator="cpu",
        devices = [device],
        max_epochs=args.n_epochs,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run = args.debug if args.debug else False,

    )

    # Train
    # trainer.fit(model, dm_any,
    #             ckpt_path=args.checkpoint if args.checkpoint else None
    #             )
    
    # trainer = Trainer(
    #     accelerator=accelerator,
    #     # accelerator="cpu",
    #     devices = [device],
    #     max_epochs=int(0.3*args.n_epochs),
    #     precision=args.precision,
    #     log_every_n_steps=args.log_every_n_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     fast_dev_run = args.debug if args.debug else False,

    # )
    
    trainer.fit(model, dm_all,
                ckpt_path=args.checkpoint if args.checkpoint else None)

    # Test
    # trainer.test(model, dm_all)

    # Save last checkpoint at the logging directory
    if args.log:
        trainer.save_checkpoint(logger.experiment.dir + "/last.ckpt")


def main():

    from utils.config import SLConfig



    parser = ArgumentParser()

    parser.add_argument("--source", type=str, default="./folds.csv")
    parser.add_argument("--cfg", type=str, default="./configs/data.json")
    parser.add_argument("--errors", type=str, default="./errors/reconstruction_errors.csv")

    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--debug", type=int, default=1)

    # Training parameters
    parser.add_argument("--model", type=str, default="spine-dino")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=16)
    # parser.add_argument("--lr", type=float, default=None)
    # parser.add_argument("--lr_backbone", type=float, default=None)
    # parser.add_argument("--momentum", type=float, default=None)
    # parser.add_argument("--weight_decay", type=float, default=None)
    # parser.add_argument("--height", type=int, default=750)
    # parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.5)
    # parser.add_argument("--bbox_expansion", type=float, default=0.4)
    parser.add_argument("--bbox_format", type=str, default="cxcywh")
    # parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_keypoints", type=int, default=6)
    parser.add_argument("--n_vertebrae", type=int, default=13)
    parser.add_argument("--n_dims", type=int, default=2)
    parser.add_argument("--missing_weight", type=float, default=1e-3)

    parser.add_argument("--p_augmentation", type=float, default=0.5)
    parser.add_argument("--fill_value", type=float, default=0)
    parser.add_argument("--filter", type=str, default="not_all")
    parser.add_argument("--polynomial_loss_coeff", type=float, default=1.0)

    # Training details
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--log_every_n_steps", type=int, default=5)
    
    # Logging
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--name", type=str, default="superb_spine")
    parser.add_argument("--n_imgs_to_log", type=int, default=4)
    parser.add_argument("--log_imgs_every_n_epochs", type=int, default=1)

    parser.add_argument("--checkpoint", type=str, default="")


    parser.add_argument('--config_file', '-c', type=str, required=True)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    # Get arguments
    args = parser.parse_args()
    
    # If a config file is provided, load it
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()

    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    # print(args_vars)

    # Train
    train(args)


if __name__ == "__main__":

    main()