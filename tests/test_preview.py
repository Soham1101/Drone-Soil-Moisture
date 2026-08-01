import io

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from pipeline.preview import orthophoto_to_png, vwc_map_to_png
from pipeline.problem_areas import (
    DRY_HIGHLIGHT_COLOR,
    WET_HIGHLIGHT_COLOR,
    compute_dry_wet_thresholds,
)


def _write_raster(path, count):
    transform = from_origin(-79.0, 36.0, 0.001, 0.001)
    is_rgb = count >= 3
    data = np.random.default_rng(0).uniform(0, 255, (count, 10, 10)).astype("uint8" if is_rgb else "float32")
    nodata = None if is_rgb else -9999.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=count,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_orthophoto_to_png_handles_rgb(tmp_path):
    path = tmp_path / "rgb.tif"
    _write_raster(path, count=4)
    png_bytes = orthophoto_to_png(str(path))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_orthophoto_to_png_handles_single_band(tmp_path):
    path = tmp_path / "single.tif"
    _write_raster(path, count=1)
    png_bytes = orthophoto_to_png(str(path))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_vwc_map_to_png_produces_valid_png(tmp_path):
    path = tmp_path / "vwc.tif"
    _write_raster(path, count=1)
    png_bytes = vwc_map_to_png(str(path))
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_orthophoto_and_vwc_map_render_at_same_size_regardless_of_source_size(tmp_path):
    # A tiny 10x10 source raster (like our small synthetic test rasters) must not
    # render smaller than the VWC map preview it's shown next to in the 2D Map tab.
    ortho_path = tmp_path / "ortho.tif"
    vwc_path = tmp_path / "vwc.tif"
    _write_raster(ortho_path, count=3)
    _write_raster(vwc_path, count=1)

    ortho_png = orthophoto_to_png(str(ortho_path), max_size=900)
    vwc_png = vwc_map_to_png(str(vwc_path), max_size=900)

    ortho_size = Image.open(io.BytesIO(ortho_png)).size
    vwc_size = Image.open(io.BytesIO(vwc_png)).size

    assert ortho_size == vwc_size == (900, 900)


def test_vwc_map_to_png_highlights_dry_and_wet_pixels(tmp_path):
    path = tmp_path / "vwc.tif"
    transform = from_origin(-79.0, 36.0, 0.001, 0.001)
    data = np.linspace(10, 40, 100, dtype="float32").reshape(10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    dry_value, wet_value = compute_dry_wet_thresholds(str(path), dry_percentile=10, wet_percentile=90)
    png_bytes = vwc_map_to_png(str(path), max_size=10, dry_value=dry_value, wet_value=wet_value)
    img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

    assert tuple(img[0, 0]) == tuple(DRY_HIGHLIGHT_COLOR)
    assert tuple(img[9, 9]) == tuple(WET_HIGHLIGHT_COLOR)
    assert tuple(img[5, 5]) not in (tuple(DRY_HIGHLIGHT_COLOR), tuple(WET_HIGHLIGHT_COLOR))
