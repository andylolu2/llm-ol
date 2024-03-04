from collections.abc import Sized
from time import time
from typing import Iterable, TypeVar

from absl import logging

T = TypeVar("T")


def textqdm(
    iterable: Iterable[T],
    total: int | None = None,
    period: float = 10.0,
) -> Iterable[T]:
    """A text-based version of tqdm."""
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)

    last_time = time() - period
    for i, item in enumerate(iterable):
        if time() - last_time > period:
            last_time = time()
            if total is not None:
                logging.info("Progress: %d / %d", i, total)
            else:
                logging.info("Progress: %d", i)
        yield item
