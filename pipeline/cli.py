"""Single CLI entrypoint for the soil moisture pipeline.

Usage:
    python -m pipeline.cli run --video flight.mp4 --sensor thermal \
        --tdr-csv readings.csv --gcp-file gcps.txt \
        --site-lat 35.71 --site-lon -78.85 --date 2026-08-15 --out outputs/run1

Individual stages (extract, map, sample, weather, model) are also exposed for
running/debugging one step at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
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


@click.group()
def cli():
    pass


@cli.command("extract")
@click.option("--video", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--interval", default=2.0, type=float, help="Seconds between extracted frames")
@click.option("--sensor", default="thermal", type=click.Choice(["thermal", "rgb"]))
def extract_cmd(video, out, interval, sensor):
    frames = extract_frames(video, out, interval_s=interval, sensor_type=sensor)
    click.echo(f"Extracted {len(frames)} frames to {out}")


@cli.command("map")
@click.option("--images", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--gcp-file", default=None, type=click.Path(exists=True))
def map_cmd(images, out, gcp_file):
    orthophoto = build_orthomosaic(images, out, gcp_file=gcp_file)
    click.echo(f"Orthomosaic written to {orthophoto}")


@cli.command("sample")
@click.option("--raster", required=True, type=click.Path(exists=True))
@click.option("--tdr-csv", required=True, type=click.Path(exists=True))
@click.option("--out-csv", required=True, type=click.Path())
def sample_cmd(raster, tdr_csv, out_csv):
    df = pd.read_csv(tdr_csv)
    result = sample_raster_at_points(raster, df)
    result.to_csv(out_csv, index=False)
    click.echo(f"Sampled values written to {out_csv}")


@cli.command("weather")
@click.option("--lat", required=True, type=float)
@click.option("--lon", required=True, type=float)
@click.option("--date", required=True, type=str)
def weather_cmd(lat, lon, date):
    result = get_weather(lat, lon, date)
    click.echo(json.dumps(result, indent=2))


@cli.command("model")
@click.option("--training-csv", required=True, type=click.Path(exists=True))
@click.option("--feature", "features", multiple=True, required=True)
@click.option("--target", default="vwc_percent")
@click.option("--out-json", required=True, type=click.Path())
def model_cmd(training_csv, features, target, out_json):
    df = pd.read_csv(training_csv)
    result = leave_one_out_cv(df, list(features), target)
    Path(out_json).write_text(json.dumps(result, indent=2))
    click.echo(f"RMSE={result['rmse']:.3f} MAE={result['mae']:.3f} -> {out_json}")


def _build_training_table(tdr_csv: str, orthophoto: str, weather: dict) -> pd.DataFrame:
    df = pd.read_csv(tdr_csv)
    df = sample_raster_at_points(orthophoto, df)
    df = df.rename(columns={"raster_value": "lst_value"})
    df = df.dropna(subset=["lst_value"])
    for key in WEATHER_FEATURE_KEYS:
        df[key] = weather[key]
    return df


@cli.command("run")
@click.option("--video", required=True, type=click.Path(exists=True))
@click.option("--sensor", default="thermal", type=click.Choice(["thermal", "rgb"]))
@click.option("--interval", default=2.0, type=float)
@click.option("--tdr-csv", required=True, type=click.Path(exists=True))
@click.option("--gcp-file", default=None, type=click.Path(exists=True))
@click.option("--site-lat", required=True, type=float)
@click.option("--site-lon", required=True, type=float)
@click.option("--date", required=True, type=str)
@click.option("--out", required=True, type=click.Path())
@click.option(
    "--orthomosaic",
    "orthomosaic_override",
    default=None,
    type=click.Path(exists=True),
    help="Skip frame extraction + ODM and use an already-built orthophoto GeoTIFF instead.",
)
def run_cmd(video, sensor, interval, tdr_csv, gcp_file, site_lat, site_lon, date, out, orthomosaic_override):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if orthomosaic_override:
        orthophoto = orthomosaic_override
        click.echo(f"Using existing orthomosaic: {orthophoto}")
    else:
        frames_dir = out_dir / "frames"
        extract_frames(video, str(frames_dir), interval_s=interval, sensor_type=sensor)
        click.echo(f"Frames extracted to {frames_dir}")

        odm_dir = out_dir / "odm"
        orthophoto = build_orthomosaic(str(frames_dir), str(odm_dir), gcp_file=gcp_file)
        click.echo(f"Orthomosaic built: {orthophoto}")

    weather = get_weather(site_lat, site_lon, date)
    click.echo(f"Weather for {date} at ({site_lat}, {site_lon}): {weather}")

    training_df = _build_training_table(tdr_csv, orthophoto, weather)
    training_csv_path = out_dir / "training_table.csv"
    training_df.to_csv(training_csv_path, index=False)

    feature_cols = ["lst_value"] + [
        k for k in WEATHER_FEATURE_KEYS if weather[k] is not None
    ]
    cv_result = leave_one_out_cv(training_df, feature_cols, "vwc_percent")
    click.echo(f"Leave-one-out CV: RMSE={cv_result['rmse']:.3f} MAE={cv_result['mae']:.3f}")

    extra_features = {k: weather[k] for k in feature_cols if k != "lst_value"}
    map_path = out_dir / "vwc_prediction_map.tif"
    predict_raster(
        training_df,
        feature_cols,
        "vwc_percent",
        orthophoto,
        extra_features,
        str(map_path),
        lst_feature_col="lst_value",
    )
    click.echo(f"VWC prediction map: {map_path}")

    report = {
        "date": date,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "feature_cols": feature_cols,
        "n_training_points": len(training_df),
        "rmse": cv_result["rmse"],
        "mae": cv_result["mae"],
        "weather": weather,
        "vwc_prediction_map": str(map_path),
    }
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    click.echo(f"Validation report: {report_path}")


if __name__ == "__main__":
    cli()
