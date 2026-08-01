"""Sample raster (e.g. thermal orthomosaic / LST) values at TDR probe coordinates."""

from __future__ import annotations

import pandas as pd
import rasterio


def sample_raster_at_points(raster_path: str, points_df: pd.DataFrame) -> pd.DataFrame:
    """Return `points_df` with a `raster_value` column added, sampled from
    `raster_path` at each row's (lat, lon).

    `points_df` must have `lat` and `lon` columns in the raster's CRS lat/lon
    (WGS84). Points falling outside the raster extent get `raster_value = None`.
    """
    with rasterio.open(raster_path) as src:
        coords = list(zip(points_df["lon"], points_df["lat"]))
        band = 1
        values = []
        for val in src.sample(coords, indexes=band):
            v = float(val[0])
            values.append(None if v == src.nodata else v)

    result = points_df.copy()
    result["raster_value"] = values
    return result
