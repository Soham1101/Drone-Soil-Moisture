"""RGB-based crop health / germination proxy, for non-thermal (e.g. Avata) footage.

Consumer RGB cameras have no near-infrared band, so true NDVI isn't possible.
Excess Green Index (ExG = 2G - R - B, computed on normalized chromaticity
coordinates to reduce lighting variation) is a well-established proxy for
live vegetation vs. bare/sparse soil in RGB-only imagery -- useful for
flagging patches that haven't germinated, not for absolute vegetation vigor.
"""

from __future__ import annotations

import io

import numpy as np
import rasterio
from PIL import Image

SPARSE_COLOR = np.array([214, 174, 116])
HEALTHY_COLOR = np.array([40, 160, 60])
BARE_HIGHLIGHT_COLOR = np.array([220, 40, 40])
NODATA_COLOR = np.array([40, 40, 40])


def compute_exg(rgb_raster_path: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return (exg_array, valid_mask, rasterio_profile) for `rgb_raster_path`."""
    with rasterio.open(rgb_raster_path) as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)
        nodata = src.nodata
        profile = src.profile.copy()

    valid_mask = None
    if nodata is not None:
        valid_mask = ~((r == nodata) & (g == nodata) & (b == nodata))

    total = r + g + b
    total[total == 0] = 1.0
    rn, gn, bn = r / total, g / total, b / total
    exg = 2 * gn - rn - bn

    return exg, valid_mask, profile


def classify_bare_soil(
    exg: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold_percentile: float = 30.0,
) -> tuple[np.ndarray, float, float]:
    """Return (bare_mask, bare_fraction, threshold_value).

    Cells with ExG at or below the given percentile of the field's own
    distribution are flagged as bare/sparse -- likely non-germinated patches.
    """
    values = exg[valid_mask] if valid_mask is not None else exg.ravel()
    threshold_value = float(np.percentile(values, threshold_percentile))

    bare_mask = exg <= threshold_value
    if valid_mask is not None:
        bare_mask = bare_mask & valid_mask
        denom = max(int(valid_mask.sum()), 1)
    else:
        denom = bare_mask.size

    bare_fraction = float(bare_mask.sum()) / denom
    return bare_mask, bare_fraction, threshold_value


def vegetation_health_to_png(
    rgb_raster_path: str,
    threshold_percentile: float = 30.0,
    max_size: int = 900,
) -> tuple[bytes, float]:
    """Render a dry(tan)-to-healthy(green) map with bare/sparse cells highlighted
    in red. Returns (png_bytes, bare_fraction).
    """
    exg, valid_mask, _ = compute_exg(rgb_raster_path)
    bare_mask, bare_fraction, _ = classify_bare_soil(exg, valid_mask, threshold_percentile)

    values = exg[valid_mask] if valid_mask is not None else exg
    lo, hi = values.min(), values.max()
    span = hi - lo if hi > lo else 1.0
    norm = np.clip((exg - lo) / span, 0, 1)

    rgb = (
        SPARSE_COLOR[None, None, :] * (1 - norm[..., None]) + HEALTHY_COLOR[None, None, :] * norm[..., None]
    ).astype("uint8")
    rgb[bare_mask] = BARE_HIGHLIGHT_COLOR
    if valid_mask is not None:
        rgb[~valid_mask] = NODATA_COLOR

    img = Image.fromarray(rgb).resize((max_size, max_size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), bare_fraction
