"""Shared pipeline orchestration, used by both cli.py and the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from pipeline.frame_extraction import extract_frames
from pipeline.model import leave_one_out_cv, predict_raster
from pipeline.orthomosaic import build_orthomosaic
from pipeline.sampling import sample_raster_at_points
from pipeline.weather import get_weather

WEATHER_FEATURE_KEYS = [
    "precipitation_sum_lookback_mm",
    "era5_soil_moisture_0_7cm_reference",
]

ProgressCallback = Optional[Callable[[str], None]]


def _report(progress_cb: ProgressCallback, message: str) -> None:
    if progress_cb:
        progress_cb(message)


def _build_training_table(tdr_csv: str, orthophoto: str, weather: dict) -> pd.DataFrame:
    df = pd.read_csv(tdr_csv)
    df = sample_raster_at_points(orthophoto, df)
    df = df.rename(columns={"raster_value": "lst_value"})
    df = df.dropna(subset=["lst_value"])
    for key in WEATHER_FEATURE_KEYS:
        df[key] = weather[key]
    return df


def run_pipeline(
    tdr_csv: str,
    site_lat: float,
    site_lon: float,
    date: str,
    out_dir: str,
    video: str | None = None,
    sensor: str = "thermal",
    interval: float = 2.0,
    gcp_file: str | None = None,
    orthomosaic_override: str | None = None,
    progress_cb: ProgressCallback = None,
) -> dict:
    """Run the full soil-moisture pipeline, returning the validation report dict.

    Either `video` (to extract frames + build a new orthomosaic) or
    `orthomosaic_override` (to reuse an existing one) must be provided.
    """
    if not video and not orthomosaic_override:
        raise ValueError("Either video or orthomosaic_override must be provided")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if orthomosaic_override:
        orthophoto = orthomosaic_override
        _report(progress_cb, f"Using existing orthomosaic: {orthophoto}")
    else:
        frames_dir = out_path / "frames"
        _report(progress_cb, "Extracting frames from video...")
        extract_frames(video, str(frames_dir), interval_s=interval, sensor_type=sensor)
        _report(progress_cb, f"Frames extracted to {frames_dir}")

        odm_dir = out_path / "odm"
        _report(progress_cb, "Building orthomosaic with OpenDroneMap (this can take several minutes)...")
        orthophoto = build_orthomosaic(str(frames_dir), str(odm_dir), gcp_file=gcp_file)
        _report(progress_cb, f"Orthomosaic built: {orthophoto}")

    _report(progress_cb, f"Fetching weather for {date} at ({site_lat}, {site_lon})...")
    weather = get_weather(site_lat, site_lon, date)
    _report(progress_cb, f"Weather: {weather}")

    _report(progress_cb, "Sampling LST at TDR probe locations...")
    training_df = _build_training_table(tdr_csv, orthophoto, weather)
    training_csv_path = out_path / "training_table.csv"
    training_df.to_csv(training_csv_path, index=False)

    feature_cols = ["lst_value"] + [k for k in WEATHER_FEATURE_KEYS if weather[k] is not None]

    _report(progress_cb, "Fitting regression-kriging model and running leave-one-out CV...")
    cv_result = leave_one_out_cv(training_df, feature_cols, "vwc_percent")
    _report(progress_cb, f"Leave-one-out CV: RMSE={cv_result['rmse']:.3f} MAE={cv_result['mae']:.3f}")

    extra_features = {k: weather[k] for k in feature_cols if k != "lst_value"}
    map_path = out_path / "vwc_prediction_map.tif"
    _report(progress_cb, "Predicting continuous VWC map...")
    predict_raster(
        training_df,
        feature_cols,
        "vwc_percent",
        orthophoto,
        extra_features,
        str(map_path),
        lst_feature_col="lst_value",
    )
    _report(progress_cb, f"VWC prediction map: {map_path}")

    report = {
        "date": date,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "feature_cols": feature_cols,
        "n_training_points": len(training_df),
        "rmse": cv_result["rmse"],
        "mae": cv_result["mae"],
        "weather": weather,
        "orthomosaic": str(orthophoto),
        "vwc_prediction_map": str(map_path),
        "training_table": str(training_csv_path),
    }
    report_path = out_path / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    report["validation_report"] = str(report_path)
    _report(progress_cb, f"Validation report: {report_path}")

    return report
