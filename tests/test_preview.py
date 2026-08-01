import numpy as np
import rasterio
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
