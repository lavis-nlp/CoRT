import random
from pathlib import Path
import argparse
import logging
import os

from pytorch_lightning import Trainer, LightningModule, seed_everything
import torch

from cort.utils import init_root_logger, type_int_or_float, type_bool
from cort.model import CortModel, TSVDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Script for training CoRT",
    )
    parser = CortModel.add_model_specific_args(parser)
    parser = TSVDataModule.add_data_specific_args(parser)
    parser.add_argument("--num_epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument(
        "--accumulate_grad_batches",
        "--update_step",
        dest="accumulate_grad_batches",
        help="Number of batch gradients accumulated per update step",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Target device(s) in Lightning format. For Instance: "4" for cuda:4. '
        "Multi-GPU training is currently not supported, though",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for all random generators", default=42
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=10000,
        help="number of training batches between validation epochs",
    )
    parser.add_argument(
        "--metric_logger",
        choices=["wandb", "tensorboard", "none"],
        default="tensorboard",
    )
    parser.add_argument(
        "--run_id", type=str, default=None, help="A unique ID for this run"
    )
    parser.add_argument("--project_name", type=str, default="cort")
    parser.add_argument("--wandb_console", type=type_bool, default=False)
    parser.add_argument("--limit_train_batches", type=type_int_or_float, default=1.0)
    parser.add_argument(
        "--max_steps", type=int, default=None, help="useful for debugging/testing"
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="Path to the checkpoint file (.ckpt) to load the model weights from",
    )
    parser.add_argument("--validate", type=type_bool, default=True)
    parser.add_argument(
        "--fit",
        type=type_bool,
        default=True,
        help="Should Training be performed? Useful if testing on checkpoint",
    )
    parser.add_argument("--test", type=type_bool, default=True)
    parser.add_argument("--ignore_warnings", type=type_bool, default=True)

    args = parser.parse_args()
    validate_args(args)
    # generate run_id if not given
    if not args.run_id:
        args.run_id = "".join(
            random.choices("0123456789abcdefghijklmnopqrstuvwxyz", k=8)
        )
    return args


def main():
    args = parse_args()
    init_root_logger(file=Path(f"./runs/{args.project_name}/{args.run_id}/train.log"),
                     loglevel=logging.INFO)
    if args.ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    model = init_model(args)
    datamodule = TSVDataModule(args, setup_val=args.validate)
    metric_logger = config_metric_logger(args)
    trainer = Trainer(
        max_epochs=args.num_epochs,
        gpus=args.device,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=metric_logger,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=1.0 if args.validate else 0.0,
        max_steps=args.max_steps,
        default_root_dir=Path("./runs").resolve(),
    )
    if args.fit:
        trainer.checkpoint_callback.save_last = True
        trainer.fit(model, datamodule=datamodule)
    if args.test:
        trainer.test(model, datamodule=datamodule)


def init_model(args):
    pl_module: LightningModule = CortModel(args)
    if args.load_checkpoint:
        logging.info("loading checkpoint")
        checkpoint = torch.load(args.load_checkpoint)
        pl_module.load_state_dict(checkpoint["state_dict"])
    return pl_module


def config_metric_logger(args):
    if args.metric_logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        # turn off wanbd console to avoid progress bar logging
        os.environ["WANDB_CONSOLE"] = "on" if args.wandb_console else "off"
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_id,
            version=args.run_id,
            save_dir=Path("./runs").resolve(),
        )
    elif args.metric_logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger(
            save_dir=Path("./runs").resolve(),
            version=args.run_id,
            name=args.project_name,
        )
    else:
        logger = None
    return logger


def validate_args(args):
    if not args.fit and not args.test:
        raise argparse.ArgumentError("Nothing to do here")
    if args.use_test_index and (args.fit or not args.load_checkpoint):
        raise argparse.ArgumentError(
            "If 'use_test_index' is set, 'fit' should be false and "
            "the corresponding checkpoint should be given."
        )
    if args.fit and args.validate:
        for p in (args.val_qrel_file, args.val_query_file, args.val_negrank_file):
            if not Path(p).exists():
                logging.warning(
                    "validation file '{}' not found, disabling validation".format(p)
                )
                args.validate = False
                break


if __name__ == "__main__":
    main()
