import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pipeline.model import RegressionKrigingModel, leave_one_out_cv, predict_raster


def _synthetic_points(n=12, seed=0):
    rng = np.random.default_rng(seed)
    lats = 36.0 + rng.uniform(-0.01, 0.01, n)
    lons = -79.0 + rng.uniform(-0.01, 0.01, n)
    lst = rng.uniform(20, 35, n)
    noise = rng.normal(0, 0.5, n)
    vwc = 40 - 0.6 * lst + noise
    return pd.DataFrame({"lat": lats, "lon": lons, "lst_value": lst, "vwc_percent": vwc})


def test_regression_kriging_recovers_known_trend():
    df = _synthetic_points()
    model = RegressionKrigingModel(["lst_value"]).fit(df, "vwc_percent")
    preds = model.predict(df)
    assert np.corrcoef(preds, df["vwc_percent"])[0, 1] > 0.8


def test_leave_one_out_cv_reports_reasonable_error():
    df = _synthetic_points(n=15, seed=1)
    result = leave_one_out_cv(df, ["lst_value"], "vwc_percent")
    assert result["rmse"] < 5.0
    assert len(result["predictions"]) == 15


def test_predict_raster_writes_valid_geotiff(tmp_path):
    df = _synthetic_points(n=10, seed=2)
    lst_raster_path = tmp_path / "lst.tif"
    transform = from_origin(-79.01, 36.01, 0.001, 0.001)
    lst_grid = np.linspace(20, 35, 8 * 8, dtype="float32").reshape(8, 8)
    with rasterio.open(
        lst_raster_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(lst_grid, 1)

    out_path = tmp_path / "vwc_map.tif"
    predict_raster(
        df,
        ["lst_value"],
        "vwc_percent",
        str(lst_raster_path),
        {},
        str(out_path),
        lst_feature_col="lst_value",
    )

    with rasterio.open(out_path) as src:
        result = src.read(1)
        assert src.crs is not None
        assert result.shape == (8, 8)
        assert np.any(result != -9999.0)
