"""Logging helpers for training runs.

This module centralizes three things:
1. normalizing logging-related CLI arguments,
2. creating a per-run log directory plus a config snapshot,
3. building and safely shutting down the logger used by `src/run.py`.
"""

import json
import logging
import os
import re

try:
    from ..dataset.loader import resolve_dataset_name
    from ..model.modalities import parse_active_modules
except ImportError:
    from dataset.loader import resolve_dataset_name
    from model.modalities import parse_active_modules


def _safe_log_name_part(value):
    """Return a filesystem-friendly component for a run log directory name."""
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe_value.strip("_") or "unknown"


def prepare_logging_args(args, repo_root, start_time):
    """Normalize logging arguments in-place before the logger is created.

    `log_root` is converted to an absolute path relative to the repository root
    so downstream code can write logs without caring how the user provided it.
    If `log_name` is empty, generate a default name in the form
    `dataset-modality-timestamp` to keep every run's outputs together.
    """
    if not isinstance(getattr(args, "log_root", None), str):
        raise TypeError("log_root must be provided as a string on args")
    if not isinstance(getattr(args, "dataset", None), str):
        raise TypeError("dataset must be provided as a string on args")
    if not isinstance(getattr(args, "modules", None), str):
        raise TypeError("modules must be provided as a string on args")

    if os.path.isabs(args.log_root):
        args.log_root = args.log_root
    else:
        args.log_root = os.path.abspath(os.path.join(repo_root, args.log_root))

    if not args.log_name:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        modules = parse_active_modules(args.modules)
        module_tag = "_".join(_safe_log_name_part(module) for module in modules) or "none"
        dataset_name = resolve_dataset_name(args.dataset)
        safe_dataset = _safe_log_name_part(dataset_name)
        args.log_name = "{}-{}-{}".format(safe_dataset, module_tag, timestamp)

    return args


def setup_training_logger(args, repo_root, start_time):
    """Create the run logger and persist the resolved runtime configuration.

    Returns the logger together with the main output paths so the caller can
    report them and reuse them elsewhere in the run lifecycle.
    """
    args = prepare_logging_args(args, repo_root, start_time)
    log_dir = os.path.join(args.log_root, args.log_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train.log')
    config_file_path = os.path.join(log_dir, 'config.json')

    # Persist the resolved arguments once at startup so each log directory is a
    # self-contained record of how that run was launched.
    with open(config_file_path, 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, ensure_ascii=False, indent=2, sort_keys=True)

    logger = logging.getLogger("run_0_2.{}".format(args.log_name))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Clear stale handlers in case the same logger name is reused within one
    # process, which would otherwise duplicate every log line.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_dir, log_file_path, config_file_path


def shutdown_logger(logger):
    """Close and detach all handlers owned by the given logger."""
    if logger is None:
        return
    # Iterate over a copy because removing handlers mutates `logger.handlers`.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
