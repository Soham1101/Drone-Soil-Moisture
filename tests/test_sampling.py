import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pipeline.sampling import sample_raster_at_points


def _write_test_raster(path, nodata=-9999.0):
    # 4x4 grid, top-left origin at (lon=-79.0, lat=36.0), 0.001 deg pixels.
    transform = from_origin(-79.0, 36.0, 0.001, 0.001)
    data = np.arange(16, dtype="float32").reshape(4, 4)
    data[0, 0] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return data


def test_sample_raster_at_points_matches_expected_pixel(tmp_path):
    raster_path = tmp_path / "test.tif"
    data = _write_test_raster(raster_path)

    points = pd.DataFrame(
        {
            "probe_id": ["p1", "p2"],
            "lon": [-78.9985, -78.9975],
            "lat": [35.9985, 35.9975],
        }
    )
    result = sample_raster_at_points(str(raster_path), points)

    assert list(result["raster_value"]) == [float(data[1, 1]), float(data[2, 2])]


def test_sample_raster_at_points_returns_none_for_nodata(tmp_path):
    raster_path = tmp_path / "test.tif"
    _write_test_raster(raster_path)

    points = pd.DataFrame({"probe_id": ["p1"], "lon": [-78.9995], "lat": [35.9995]})
    result = sample_raster_at_points(str(raster_path), points)

    assert result["raster_value"].iloc[0] is None
