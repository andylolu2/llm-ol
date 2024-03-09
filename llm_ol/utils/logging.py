from pathlib import Path

from absl import logging


def setup_logging(dir: Path | str, log_file_name: str = "logging"):
    log_dir = Path(dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Set up logging
    logging.get_absl_handler().use_absl_log_file(log_file_name, str(log_dir))
    logging.set_stderrthreshold("info")
    logging.set_verbosity("debug")
    print(f"Logging to {log_dir}")
