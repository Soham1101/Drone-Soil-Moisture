"""Sample raster (e.g. thermal orthomosaic / LST) values at TDR probe coordinates."""

from __future__ import annotations

import pandas as pd
import rasterio
from pyproj import Transformer


def sample_raster_at_points(raster_path: str, points_df: pd.DataFrame) -> pd.DataFrame:
    """Return `points_df` with a `raster_value` column added, sampled from
    `raster_path` at each row's (lat, lon).

    `points_df`'s `lat`/`lon` columns are always WGS84 (EPSG:4326), matching
    the TDR CSV format -- reprojected here into the raster's own CRS before
    sampling, since a real ODM orthomosaic is typically in a local UTM zone,
    not WGS84. Points falling outside the raster extent get `raster_value = None`.
    """
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(points_df["lon"].to_numpy(), points_df["lat"].to_numpy())
        else:
            xs, ys = points_df["lon"].to_numpy(), points_df["lat"].to_numpy()

        coords = list(zip(xs, ys))
        band = 1
        values = []
        for val in src.sample(coords, indexes=band):
            v = float(val[0])
            values.append(None if v == src.nodata else v)

    result = points_df.copy()
    result["raster_value"] = values
    return result
