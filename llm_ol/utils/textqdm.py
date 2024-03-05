from collections.abc import Sized
from time import time
from typing import Iterable, TypeVar

from absl import logging

T = TypeVar("T")


class textpbar:
    """A text-based version of a progress bar."""

    def __init__(self, total: int | None = None, period: float = 10.0):
        self.total = total
        self.period = period
        self.start_time = time()
        self.last_time = time() - period
        self.i = 0
        self.update(0)

    def update(self, n: int = 1):
        self.i += n
        if time() - self.last_time > self.period:
            self.last_time = time()
            avg_rate = self.i / (time() - self.start_time)
            if self.total is not None:
                logging.info(
                    "Progress: %d / %d (Avg. rate: %.2f it/s)",
                    self.i,
                    self.total,
                    avg_rate,
                )
            else:
                logging.info("Progress: %d (Avg. rate: %.2f it/s)", self.i, avg_rate)


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
