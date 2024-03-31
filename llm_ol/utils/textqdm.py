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
        running_avg_rate: float = 0.9,
    ):
        self.total = total
        self.period = period
        self.running_avg_rate = running_avg_rate
        self.last_time = time() - period
        self.last_i = 0
        self.avg_rate = 0
        self.i = 0
        self.update(0)

    def update(self, n: int = 1):
        self.i += n
        if time() - self.last_time > self.period:
            di = self.i - self.last_i
            dt = time() - self.last_time
            self.avg_rate = (
                self.running_avg_rate * self.avg_rate
                + (1 - self.running_avg_rate) * di / dt
            )

            self.last_time = time()
            self.last_i = self.i

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
