"""Threshold-based dry/wet problem-area detection on a VWC prediction raster.

Thresholds are computed once from the full-resolution raster so the 2D map,
3D view, and summary stats all agree on the same absolute VWC values, even
though the 3D view only samples a coarse grid of that raster.
"""

from __future__ import annotations

import numpy as np
import rasterio

DRY_HIGHLIGHT_COLOR = np.array([220, 40, 40])
WET_HIGHLIGHT_COLOR = np.array([90, 50, 200])


def compute_dry_wet_thresholds(
    raster_path: str, dry_percentile: float, wet_percentile: float
) -> tuple[float, float]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata
    valid = arr[arr != nodata] if nodata is not None else arr.ravel()
    dry_value = float(np.percentile(valid, dry_percentile))
    wet_value = float(np.percentile(valid, wet_percentile))
    return dry_value, wet_value


def problem_area_fractions(raster_path: str, dry_value: float, wet_value: float) -> tuple[float, float]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    valid_mask = arr != nodata if nodata is not None else np.ones_like(arr, dtype=bool)
    denom = max(int(valid_mask.sum()), 1)
    dry_fraction = float(((arr <= dry_value) & valid_mask).sum()) / denom
    wet_fraction = float(((arr >= wet_value) & valid_mask).sum()) / denom
    return dry_fraction, wet_fraction
