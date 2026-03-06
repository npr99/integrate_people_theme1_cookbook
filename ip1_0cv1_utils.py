"""
Shared utilities for data processing and naming.
"""

import os


def normalize_resolution(resolution):
    """Normalize resolution names to the uifl_* pattern."""
    if resolution is None:
        raise ValueError("Resolution cannot be None")

    resolution_str = str(resolution).strip()
    if resolution_str in {"1km", "4km"}:
        return f"uifl_{resolution_str}"
    if resolution_str.startswith("uifl_"):
        return resolution_str
    return resolution_str


def format_grid_id(resolution, row, col):
    """Create a standardized grid cell id for merging."""
    normalized = normalize_resolution(resolution)
    return f"air{normalized}_{int(row):04d}_{int(col):04d}"


def ensure_dir(path):
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
