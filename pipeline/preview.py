"""Render orthophoto / VWC map GeoTIFFs as PNG bytes for quick visual display."""

from __future__ import annotations

import io

import numpy as np
import rasterio
from PIL import Image

DRY_COLOR = np.array([214, 174, 116])
WET_COLOR = np.array([30, 90, 180])
NODATA_COLOR = np.array([40, 40, 40])


def orthophoto_to_png(raster_path: str, max_size: int = 900) -> bytes:
    with rasterio.open(raster_path) as src:
        if src.count >= 3:
            arr = src.read([1, 2, 3])
            img = Image.fromarray(np.moveaxis(arr, 0, -1))
        else:
            band = src.read(1).astype(float)
            lo, hi = band.min(), band.max()
            span = hi - lo if hi > lo else 1.0
            normalized = ((band - lo) / span * 255).astype("uint8")
            img = Image.fromarray(normalized, mode="L")

    img = img.resize((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def vwc_map_to_png(raster_path: str, max_size: int = 900) -> bytes:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    valid = arr[arr != nodata] if nodata is not None else arr
    lo, hi = valid.min(), valid.max()
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((arr - lo) / span, 0, 1)

    rgb = (
        DRY_COLOR[None, None, :] * (1 - norm[..., None]) + WET_COLOR[None, None, :] * norm[..., None]
    ).astype("uint8")
    if nodata is not None:
        rgb[arr == nodata] = NODATA_COLOR

    img = Image.fromarray(rgb)
    img = img.resize((max_size, max_size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
