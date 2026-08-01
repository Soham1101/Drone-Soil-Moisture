"""Generate a synthetic thermal orthomosaic + TDR readings dataset for
end-to-end pipeline testing, since no real flight data exists yet.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

SAMPLE_DIR = Path(__file__).resolve().parent

# Roughly matches the field site: small strawberry farm extent (~800m square).
ORIGIN_LON, ORIGIN_LAT = -78.86, 35.72
PIXEL_SIZE_DEG = 0.0002
GRID_SIZE = 40


def make_orthomosaic(seed: int = 0) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    lst = 25 + 5 * np.sin(xx / 6) + 3 * np.cos(yy / 5) + rng.normal(0, 0.3, xx.shape)

    hidden_field = gaussian_filter(rng.normal(0, 1, xx.shape), sigma=3)
    vwc_true = 45 - 0.8 * lst + 3 * hidden_field

    transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
    lst_path = SAMPLE_DIR / "synthetic_orthomosaic.tif"
    with rasterio.open(
        lst_path,
        "w",
        driver="GTiff",
        height=GRID_SIZE,
        width=GRID_SIZE,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(lst.astype("float32"), 1)

    return lst_path, vwc_true


def make_tdr_readings(vwc_true: np.ndarray, n_probes: int = 15, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    rows = rng.integers(2, GRID_SIZE - 2, n_probes)
    cols = rng.integers(2, GRID_SIZE - 2, n_probes)

    transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
    lons, lats = rasterio.transform.xy(transform, rows, cols)

    measured_vwc = vwc_true[rows, cols] + rng.normal(0, 0.5, n_probes)

    df = pd.DataFrame(
        {
            "probe_id": [f"tdr_{i:02d}" for i in range(n_probes)],
            "lat": lats,
            "lon": lons,
            "timestamp": "2026-07-25T14:00:00",
            "vwc_percent": measured_vwc,
        }
    )
    csv_path = SAMPLE_DIR / "tdr_readings.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_veg_rgb_raster(seed: int = 2) -> Path:
    """Synthetic RGB orthomosaic with a deliberate bare/non-germinated patch,
    for demoing the crop-health (ExG) module without real Avata footage.
    """
    rng = np.random.default_rng(seed)
    healthy_rgb = np.array([60, 150, 50])
    bare_rgb = np.array([170, 130, 90])

    rgb = np.tile(healthy_rgb, (GRID_SIZE, GRID_SIZE, 1)).astype(float)
    rgb += rng.normal(0, 8, rgb.shape)

    # A rectangular non-germinated patch plus a thin bare strip.
    rgb[10:18, 5:15] = bare_rgb + rng.normal(0, 8, (8, 10, 3))
    rgb[25:28, :] = bare_rgb + rng.normal(0, 8, (3, GRID_SIZE, 3))

    rgb = np.clip(rgb, 0, 255).astype("uint8")

    transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
    veg_path = SAMPLE_DIR / "synthetic_veg_rgb.tif"
    with rasterio.open(
        veg_path,
        "w",
        driver="GTiff",
        height=GRID_SIZE,
        width=GRID_SIZE,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))

    return veg_path


def make_test_video(duration_s: int = 10, fps: int = 5) -> Path:
    video_path = SAMPLE_DIR / "synthetic_flight.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size=128x128:rate={fps}",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path


if __name__ == "__main__":
    lst_path, vwc_true = make_orthomosaic()
    csv_path = make_tdr_readings(vwc_true)
    video_path = make_test_video()
    veg_path = make_veg_rgb_raster()
    print(f"Orthomosaic: {lst_path}")
    print(f"TDR readings: {csv_path}")
    print(f"Test video: {video_path}")
    print(f"Veg RGB raster: {veg_path}")
