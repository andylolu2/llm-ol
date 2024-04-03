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
        window: int = 100,
    ):
        assert window > 0
        self.total = total
        self.period = period
        self.window = window
        self.history = []
        self.i = 0
        self.last_log = time()

    def update(self, n: int = 1):
        self.i += n
        self.history.append((self.i, time()))
        self.history = self.history[-self.window :]

        if time() - self.last_log > self.period:
            self.last_log = time()
            rate = (self.history[-1][0] - self.history[0][0]) / (
                self.history[-1][1] - self.history[0][1]
            )
            if self.total is not None:
                logging.info(
                    "Progress: %d / %d %.2f%% (Avg. rate: %.2f it/s)",
                    self.i,
                    self.total,
                    self.i / self.total * 100,
                    rate,
                )
            else:
                logging.info("Progress: %d (Avg. rate: %.2f it/s)", self.i, rate)


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
