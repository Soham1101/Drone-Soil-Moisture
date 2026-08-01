import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.vegetation import classify_bare_soil, compute_exg, vegetation_health_to_png


def _write_rgb_raster(path, nodata=None):
    transform = from_origin(-79.0, 36.0, 0.001, 0.001)
    rng = np.random.default_rng(0)
    # Half the field strongly green (healthy), half strongly brown/red (bare).
    r = np.zeros((10, 10), dtype="uint8")
    g = np.zeros((10, 10), dtype="uint8")
    b = np.zeros((10, 10), dtype="uint8")
    r[:, :5] = 180
    g[:, :5] = 60
    b[:, :5] = 40
    r[:, 5:] = 40
    g[:, 5:] = 180
    b[:, 5:] = 40
    data = np.stack([r, g, b])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_compute_exg_scores_green_higher_than_bare(tmp_path):
    path = tmp_path / "rgb.tif"
    _write_rgb_raster(path)
    exg, valid_mask, _ = compute_exg(str(path))

    assert valid_mask is None
    assert exg[:, 5:].mean() > exg[:, :5].mean()


def test_classify_bare_soil_flags_low_exg_side(tmp_path):
    path = tmp_path / "rgb.tif"
    _write_rgb_raster(path)
    exg, valid_mask, _ = compute_exg(str(path))

    bare_mask, bare_fraction, threshold = classify_bare_soil(exg, valid_mask, threshold_percentile=50.0)

    assert bare_mask[:, :5].all()
    assert not bare_mask[:, 5:].any()
    assert 0.4 < bare_fraction < 0.6


def test_vegetation_health_to_png_produces_valid_png(tmp_path):
    path = tmp_path / "rgb.tif"
    _write_rgb_raster(path)
    png_bytes, bare_fraction = vegetation_health_to_png(str(path))

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert 0.0 <= bare_fraction <= 1.0
