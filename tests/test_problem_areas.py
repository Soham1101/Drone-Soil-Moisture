import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.problem_areas import compute_dry_wet_thresholds, problem_area_fractions


def _write_vwc_raster(path, nodata=-9999.0):
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
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_compute_dry_wet_thresholds_match_percentiles(tmp_path):
    path = tmp_path / "vwc.tif"
    _write_vwc_raster(path)

    dry_value, wet_value = compute_dry_wet_thresholds(str(path), dry_percentile=10, wet_percentile=90)

    assert 10 < dry_value < 15
    assert 35 < wet_value < 40
    assert dry_value < wet_value


def test_problem_area_fractions_roughly_match_percentiles(tmp_path):
    path = tmp_path / "vwc.tif"
    _write_vwc_raster(path)

    dry_value, wet_value = compute_dry_wet_thresholds(str(path), dry_percentile=10, wet_percentile=90)
    dry_fraction, wet_fraction = problem_area_fractions(str(path), dry_value, wet_value)

    assert 0.05 <= dry_fraction <= 0.15
    assert 0.05 <= wet_fraction <= 0.15
