"""Utils module."""

import os
from pathlib import Path


def path_main_data_directory() -> Path:
    """
    Return the path to the root data directory.

    :return: path to the root data directory.
    """
    return Path(os.getenv("SCRATCH", ".."), "data").resolve()
