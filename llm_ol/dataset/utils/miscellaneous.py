from itertools import islice
from pathlib import Path
from typing import Iterable, TypeVar

from absl import logging

T = TypeVar("T")


def batch(it: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Batch an iterable into chunks of size `size`."""
    it = iter(it)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def setup_loggging(dir: Path, log_file_name: str = "logging"):
    log_dir = dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Set up logging
    logging.get_absl_handler().use_absl_log_file(log_file_name, log_dir)
    logging.set_stderrthreshold("info")
    logging.set_verbosity("debug")
