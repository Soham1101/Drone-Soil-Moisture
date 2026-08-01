"""Regression kriging: VWC ~ linear trend(LST, weather) + kriged spatial residual.

This mirrors the questionnaire's described method (regression kriging with a
thermal covariate) without depending on R/gstat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _utm_crs_for_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    hemisphere = 326 if lat >= 0 else 327
    return f"EPSG:{hemisphere}{zone:02d}"


class RegressionKrigingModel:
    def __init__(self, feature_cols: list[str], variogram_model: str = "spherical"):
        self.feature_cols = feature_cols
        self.variogram_model = variogram_model
        self.regression = LinearRegression()
        self.kriging: OrdinaryKriging | None = None
        self.target_crs: str | None = None
        self._transformer: Transformer | None = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        lat_col: str = "lat",
        lon_col: str = "lon",
        target_crs: str | None = None,
    ) -> "RegressionKrigingModel":
        X = df[self.feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        self.regression.fit(X, y)
        residuals = y - self.regression.predict(X)

        self.target_crs = target_crs or _utm_crs_for_lonlat(
            df[lon_col].mean(), df[lat_col].mean()
        )
        self._transformer = Transformer.from_crs("EPSG:4326", self.target_crs, always_xy=True)
        x, y_proj = self._transformer.transform(
            df[lon_col].to_numpy(dtype=float), df[lat_col].to_numpy(dtype=float)
        )

        self.kriging = OrdinaryKriging(
            x,
            y_proj,
            residuals,
            variogram_model=self.variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        return self

    def predict(self, df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=float)
        trend = self.regression.predict(X)
        x, y_proj = self._transformer.transform(
            df[lon_col].to_numpy(dtype=float), df[lat_col].to_numpy(dtype=float)
        )
        residual_pred, _ = self.kriging.execute("points", x, y_proj)
        return trend + np.asarray(residual_pred)


def leave_one_out_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> dict:
    n = len(df)
    preds = np.empty(n)
    for i in range(n):
        train = df.drop(df.index[i]).reset_index(drop=True)
        test = df.iloc[[i]]
        model = RegressionKrigingModel(feature_cols).fit(train, target_col, lat_col, lon_col)
        preds[i] = model.predict(test, lat_col, lon_col)[0]

    actual = df[target_col].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
        "mae": float(mean_absolute_error(actual, preds)),
        "predictions": preds.tolist(),
    }


def predict_raster(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lst_raster_path: str,
    extra_features: dict[str, float],
    out_path: str,
    lst_feature_col: str = "lst_value",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> str:
    """Fit a regression-kriging model and predict a continuous VWC raster.

    `lst_feature_col` must be the name of the per-pixel LST feature within
    `feature_cols`; every other feature is a per-flight constant supplied via
    `extra_features` (e.g. weather values), broadcast to all pixels.
    """
    with rasterio.open(lst_raster_path) as src:
        lst = src.read(1).astype(float)
        profile = src.profile.copy()
        src_nodata = src.nodata
        transform = src.transform
        rows, cols = lst.shape

    model = RegressionKrigingModel(feature_cols).fit(
        training_df, target_col, lat_col, lon_col, target_crs=str(profile["crs"])
    )

    valid_mask = lst != src_nodata if src_nodata is not None else np.ones_like(lst, dtype=bool)

    other_cols = [c for c in feature_cols if c != lst_feature_col]
    feature_matrix = np.column_stack(
        [lst.ravel()] + [np.full(lst.size, extra_features[c]) for c in other_cols]
    )
    # feature_cols order must match training; reorder columns accordingly.
    col_order = [feature_cols.index(lst_feature_col)] + [
        feature_cols.index(c) for c in other_cols
    ]
    ordered = np.empty_like(feature_matrix)
    ordered[:, col_order] = feature_matrix
    trend_flat = model.regression.predict(ordered)
    trend = trend_flat.reshape(rows, cols)

    # North-up raster assumption: x depends only on column, y only on row.
    xs = transform.c + (np.arange(cols) + 0.5) * transform.a
    ys = transform.f + (np.arange(rows) + 0.5) * transform.e
    residual_grid, _ = model.kriging.execute("grid", xs, ys)
    residual_grid = np.asarray(residual_grid)

    prediction = trend + residual_grid
    prediction[~valid_mask] = np.nan

    profile.update(dtype="float32", nodata=-9999.0, count=1)
    out_array = np.where(np.isnan(prediction), -9999.0, prediction).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_array, 1)

    return out_path
