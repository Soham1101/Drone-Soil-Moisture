"""Generate synthetic thermal orthomosaic + TDR readings datasets for
end-to-end pipeline testing, since no real flight data exists yet.

Two scenarios are generated, matching the syllabus's actual planned data
collection (dry period vs. post-rain), each with a deliberate, obvious
dry/wet hotspot so the problem-area highlighting has something clear to flag.
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

# (row, col) blob centers reused across scenarios so the same physical spots
# show up as "the dry corner" / "the wet corner" in both flights.
HOTSPOT_CENTER = (12, 28)
COLDSPOT_CENTER = (28, 10)

SCENARIOS = {
    "dry": {
        "suffix": "",
        "baseline_lst": 27.0,
        "hotspot_amplitude": 10.0,
        "coldspot_amplitude": -4.0,
        "timestamp": "2026-07-10T14:00:00",
    },
    "post_rain": {
        "suffix": "_postrain",
        "baseline_lst": 21.0,
        "hotspot_amplitude": 4.0,
        "coldspot_amplitude": -10.0,
        "timestamp": "2026-07-13T14:00:00",
    },
}


def _blob(center: tuple[int, int], amplitude: float, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return amplitude * np.exp(-r2 / (2 * sigma**2))


def make_orthomosaic(scenario: str = "dry", seed: int = 0) -> tuple[Path, np.ndarray]:
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]

    lst = (
        cfg["baseline_lst"]
        + 2 * np.sin(xx / 6)
        + 1.5 * np.cos(yy / 5)
        + rng.normal(0, 0.3, xx.shape)
        + _blob(HOTSPOT_CENTER, cfg["hotspot_amplitude"])
        + _blob(COLDSPOT_CENTER, cfg["coldspot_amplitude"])
    )

    hidden_field = gaussian_filter(rng.normal(0, 1, xx.shape), sigma=3)
    vwc_true = 45 - 0.8 * lst + 3 * hidden_field

    transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
    lst_path = SAMPLE_DIR / f"synthetic_orthomosaic{cfg['suffix']}.tif"
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


def make_tdr_readings(
    vwc_true: np.ndarray, scenario: str = "dry", n_probes: int = 20, seed: int = 1
) -> Path:
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    rows = rng.integers(2, GRID_SIZE - 2, n_probes)
    cols = rng.integers(2, GRID_SIZE - 2, n_probes)

    # Force a few probes right onto the hotspot/coldspot so the model actually
    # sees ground truth there, instead of relying on kriging to guess blindly.
    rows[0], cols[0] = HOTSPOT_CENTER
    rows[1], cols[1] = COLDSPOT_CENTER

    transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
    lons, lats = rasterio.transform.xy(transform, rows, cols)

    measured_vwc = vwc_true[rows, cols] + rng.normal(0, 0.5, n_probes)

    df = pd.DataFrame(
        {
            "probe_id": [f"tdr_{i:02d}" for i in range(n_probes)],
            "lat": lats,
            "lon": lons,
            "timestamp": cfg["timestamp"],
            "vwc_percent": measured_vwc,
        }
    )
    csv_path = SAMPLE_DIR / f"tdr_readings{cfg['suffix']}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_veg_rgb_raster(seed: int = 2) -> Path:
    """Synthetic RGB orthomosaic resembling drilled crop rows, with one row
    gap (non-germinated) and a small bare patch, for demoing the crop-health
    (ExG) module without real Avata footage.
    """
    rng = np.random.default_rng(seed)
    healthy_rgb = np.array([60, 150, 50])
    # Visually distinct from healthy canopy (darker/more shadowed, reads as
    # "rows" to the eye) but close in Excess Green Index -- true bare soil is
    # the outlier the classifier should catch, not the normal gaps between rows.
    soil_between_rows = np.array([70, 145, 55])
    bare_rgb = np.array([170, 130, 90])

    yy, xx = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    row_period = 4
    in_row = (yy % row_period) < 2  # crop rows run east-west, alternating with bare soil

    rgb = np.where(in_row[..., None], healthy_rgb, soil_between_rows).astype(float)
    rgb += rng.normal(0, 6, rgb.shape)

    # One full row that never germinated, plus a small bare patch elsewhere.
    gap_row = 16
    rgb[gap_row : gap_row + 2, :] = bare_rgb + rng.normal(0, 8, (2, GRID_SIZE, 3))
    rgb[28:33, 22:30] = bare_rgb + rng.normal(0, 8, (5, 8, 3))

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
    for scenario in SCENARIOS:
        lst_path, vwc_true = make_orthomosaic(scenario=scenario)
        csv_path = make_tdr_readings(vwc_true, scenario=scenario)
        print(f"[{scenario}] Orthomosaic: {lst_path}")
        print(f"[{scenario}] TDR readings: {csv_path}")

    video_path = make_test_video()
    veg_path = make_veg_rgb_raster()
    print(f"Test video: {video_path}")
    print(f"Veg RGB raster: {veg_path}")
