"""Path helpers shared by training and feature-encoding entrypoints."""

import os
from datetime import datetime


def build_run_dir(repo_root, dataset_name, modalities, root_name="logs"):
    """Create and return a timestamped directory under ``root_name``."""
    modality_name = "_".join(str(item) for item in modalities) or "none"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}-{}-{}".format(dataset_name, modality_name, timestamp)
    output_dir = os.path.join(repo_root, root_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
