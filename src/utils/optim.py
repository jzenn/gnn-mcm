import argparse
from typing import Iterator

import torch.nn as nn
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR

from src.utils.scheduler import CyclicalScheduler, RobbinsMonroScheduler


def get_optimizer(
    args: argparse.Namespace,
    params: Iterator[nn.Parameter],
    optimizer: str = None,
    lr: float = None,
) -> optim.Optimizer:
    lr = args.lr if lr is None else lr
    return optim.Adam(params=params, lr=lr)


def get_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer):
    warmup = args.number_epochs * args.lr_schedule_warmup_fraction
    if args.lr_schedule == "RM":
        return RobbinsMonroScheduler(args, optimizer, warmup)
    if args.lr_schedule == "Step":
        return StepLR(
            optimizer,
            step_size=args.lr_schedule_step_size,
            gamma=args.lr_schedule_step_gamma,
        )
    if args.lr_schedule == "Cyclical":
        return CyclicalScheduler(
            args,
            optimizer,
            max_learning_rate=args.lr,
            min_learning_rate=args.lr_scheduler_min_lr,
        )
    else:
        raise RuntimeError(f"Unrecognized scheduler {args.scheduler}.")
