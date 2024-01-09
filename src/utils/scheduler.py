import argparse
from math import floor
from typing import List, Tuple, Union

import torch


class RobbinsMonroScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        args: argparse.Namespace,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
    ) -> None:
        self.lr = args.lr
        self._last_lr = args.lr
        self.a = args.lr_schedule_a
        self.b = args.lr_schedule_b
        self.gamma = args.lr_schedule_rm_gamma
        self.total_steps = 0
        self.warmup_steps = warmup_steps

        super(RobbinsMonroScheduler, self).__init__(optimizer, -1)

    def get_lr(self) -> List[float]:
        lr = self.get()
        self.total_steps += 1
        return [lr]

    def get(self) -> float:
        if self.total_steps > self.warmup_steps:
            # for a = 1 at step = 0 we get the initial lr
            lr = self.lr / (
                ((self.total_steps - self.warmup_steps) / self.b + self.a) ** self.gamma
            )
        else:
            lr = self.lr
        return lr


class CyclicalScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    along the lines of
    Smith: Cyclical Learning Rates for Training Neural Networks
    """

    def __init__(
        self,
        args: argparse.Namespace,
        optimizer: torch.optim.Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
    ) -> None:
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.step_size = int(args.number_epochs / args.lr_scheduler_number_of_cycles)
        self.total_steps = 0

        super(CyclicalScheduler, self).__init__(optimizer, -1)

    def get_lr(self) -> List[float]:
        lr = self.get()
        self.total_steps += 1
        return [lr]

    def get(self) -> float:
        cycle = floor(1 + self.total_steps / (2 * self.step_size))
        x = abs(self.total_steps / self.step_size - 2 * cycle + 1)
        lr = self.min_learning_rate + (
            self.max_learning_rate - self.min_learning_rate
        ) * max(0.0, (1 - x))
        return lr


class LinearClippingScheduler:
    def __init__(
        self,
        number_of_steps: int,
        start: float,
        a: float,
    ) -> None:
        self.start = start
        self.a = a
        self.number_of_steps = number_of_steps

    def __call__(self, current_step: int) -> Tuple[Union[float, None], bool]:
        if current_step > self.number_of_steps:
            return None, False
        else:
            return current_step * self.a + self.start, True


def get_clipping_schedule(args: argparse.Namespace):
    number_of_epochs = int(args.clipping_schedule_fraction * args.number_epochs)
    return LinearClippingScheduler(
        number_of_epochs,
        args.max_grad,
        1.0 / (1.0 / args.clipping_schedule_max_grad_factor * number_of_epochs),
    )
