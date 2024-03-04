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
        self.last_time = time() - period
        self.i = 0
        self.update(0)

    def update(self, n: int = 1):
        self.i += n
        if time() - self.last_time > self.period:
            self.last_time = time()
            if self.total is not None:
                logging.info("Progress: %d / %d", self.i, self.total)
            else:
                logging.info("Progress: %d", self.i)


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
