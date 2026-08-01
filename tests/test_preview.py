import io

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from pipeline.preview import orthophoto_to_png, vwc_map_to_png


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
