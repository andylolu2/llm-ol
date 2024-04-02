from collections.abc import Sized
from time import time
from typing import Iterable, TypeVar

from absl import logging

T = TypeVar("T")


class textpbar:
    """A text-based version of a progress bar."""

    def __init__(
        self,
        total: int | None = None,
        period: float = 10.0,
        running_avg_rate: float = 0.99,
    ):
        self.total = total
        self.period = period
        self.alpha = running_avg_rate
        self.last_log = time()
        self.last_time = time()
        self.last_i = 0
        self.avg_rate = None
        self.i = 0
        self.update(0)

    def update(self, n: int = 1):
        self.i += n
        rate = (self.i - self.last_i) / (time() - self.last_time)
        self.avg_rate = (
            (self.alpha * self.avg_rate + (1 - self.alpha) * rate)
            if self.avg_rate is not None
            else rate
        )
        self.last_time = time()
        self.last_i = self.i

        if time() - self.last_log > self.period:
            self.last_log = time()
            if self.total is not None:
                logging.info(
                    "Progress: %d / %d %.2f%% (Avg. rate: %.2f it/s)",
                    self.i,
                    self.total,
                    self.i / self.total * 100,
                    self.avg_rate,
                )
            else:
                logging.info(
                    "Progress: %d (Avg. rate: %.2f it/s)", self.i, self.avg_rate
                )


def textqdm(
    iterable: Iterable[T],
    total: int | None = None,
    period: float = 10.0,
) -> Iterable[T]:
    """A text-based version of tqdm."""
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)

    pbar = textpbar(total, period)
    for item in iterable:
        pbar.update()
        yield item
