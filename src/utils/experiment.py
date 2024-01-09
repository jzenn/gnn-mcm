import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.io import load_json, save_arguments_to_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    return device


def load_wandb_file_to_environment(args: argparse.Namespace) -> Dict[str, str]:
    wandb_props = load_json(args.wandb_file)
    for k, v in wandb_props.items():
        os.environ[k] = v


def get_full_experiment_path(args: argparse.Namespace) -> Tuple[str, str]:
    current_date_time = get_datetime_str()
    experiment_run = current_date_time
    if args.slurm_job_id is not None:
        experiment_run += f"-{args.slurm_job_id}"
    path = os.path.join(args.experiment_base_path, args.experiment_name, experiment_run)
    return path, experiment_run


def get_datetime_str() -> str:
    current_date_time = datetime.today().strftime("%Y_%m_%d_%H-%M-%S-%s")
    return current_date_time


def create_path(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_full_path(args: argparse.Namespace, filename: str) -> str:
    return os.path.join(args.experiment_path, filename)


def get_summary_writer(
    args: argparse.Namespace,
) -> torch.utils.tensorboard.SummaryWriter:
    return SummaryWriter(args.tensorboard_path)


def save_numpy_arrays(args: argparse.Namespace, **kwargs: np.ndarray) -> None:
    for file_name, numpy_array in kwargs.items():
        if numpy_array is None:
            continue
        np.save(get_full_path(args, f"{file_name}.npy"), numpy_array)


def create_experiment(args: argparse.Namespace) -> None:
    # create folders along path to experiment
    path, experiment_run = get_full_experiment_path(args)
    create_path(os.path.join(path, "checkpoints"))
    # update arguments
    args.experiment_path = path
    args.tensorboard_path = os.path.join(path, "tensorboard")
    args.wandb = args.wandb and args.slurm_job_partition != "test"
    # WandB
    if args.wandb and not args.wandb_initialized:
        if args.wandb_file is None:
            raise RuntimeError("Cannot use WandB without configuration file.")
        load_wandb_file_to_environment(args)
        wandb.tensorboard.patch(root_logdir=args.tensorboard_path, pytorch=True)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.experiment_name,
        )
        # save experiment config to WandB
        wandb.config.update(args)
        # set the run name of WandB
        wandb.run.name = experiment_run
    # print and save current arguments
    print("*" * 80)
    print(f"using device {get_device().type}.")
    print("arguments for this run:")
    for argument, value in vars(args).items():
        print("\t" + str(argument) + (" " * (50 - len(str(argument)))) + str(value))
    print("*" * 80)
    save_arguments_to_path(args, os.path.join(args.experiment_path, "args.txt"))
