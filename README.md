# Thermal UAV Soil Moisture Pipeline

Estimates volumetric water content (VWC) from thermal drone footage using
regression kriging: a linear trend on land-surface-temperature (LST) +
weather covariates, plus kriged spatial residuals, trained against TDR probe
ground truth. Built for the "Soil Moisture mapped via Thermal UAV Imagery"
independent study.

## Setup

Already installed on this machine: Python 3.12, ffmpeg.

Still needed for the orthomosaic (photogrammetry) stage:

```powershell
wsl --install
# restart Windows if prompted
winget install -e --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
# launch Docker Desktop once from the Start menu, let first-run setup finish
docker version   # should succeed with no error
```

Optional, for manually picking GCP coordinates and viewing output maps:
install [QGIS](https://qgis.org/) (not required to run the pipeline itself).

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## GUI

A Streamlit app wraps the same pipeline with file uploaders, a run button,
and inline results (map preview, RMSE/MAE, downloads):

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

or double-click `run_app.bat`. It opens at http://localhost:8501. Check "Use
bundled synthetic demo data" to try it immediately with no inputs of your own.

## Running the full pipeline (CLI)

```powershell
.\.venv\Scripts\python.exe -m pipeline.cli run `
  --video path\to\flight.mp4 `
  --sensor thermal `
  --tdr-csv path\to\tdr_readings.csv `
  --gcp-file path\to\gcp_list.txt `
  --site-lat 35.71 --site-lon -78.85 --date 2026-08-15 `
  --out outputs\run1
```

This extracts frames, builds a georeferenced orthomosaic via OpenDroneMap
(Docker), samples LST at each TDR probe location, pulls weather for the site
+ date from Open-Meteo, fits a regression-kriging model, and writes:

- `outputs/run1/training_table.csv` — combined probe + LST + weather table
- `outputs/run1/vwc_prediction_map.tif` — continuous VWC prediction (GeoTIFF)
- `outputs/run1/validation_report.json` — leave-one-out CV RMSE/MAE + inputs

If you already have an orthomosaic (e.g. built separately, or for testing),
skip frame extraction/ODM with `--orthomosaic path\to\existing_ortho.tif`.

Individual stages can also be run separately: `extract`, `map`, `sample`,
`weather`, `model` (see `pipeline/cli.py --help`).

## Input file formats

- **TDR readings CSV**: see [`data/tdr_readings_template.csv`](data/tdr_readings_template.csv)
  — columns `probe_id, lat, lon, timestamp, vwc_percent`.
- **GCP file**: see [`data/gcp_template.txt`](data/gcp_template.txt) — since
  O4/Avata DVR footage has no embedded GPS, 3-5 known real-world points
  (e.g. field corners) are needed to georeference the orthomosaic. Identify
  them visually in QGIS or any image viewer.

## Known limitations / things to revisit with real data

- **Thermal values are relative, not calibrated temperature.** Unless the
  camera outputs radiometric data, video-recorded thermal frames are already
  tone-mapped for display. This matches the questionnaire's framing of
  thermal anomalies as *proxies*, but means LST values are relative within a
  flight, not directly comparable in °C across flights without a reference.
- **Weather is single-site, not per-pixel.** One manually-entered lat/lon is
  used per flight (moisture-relevant weather doesn't vary meaningfully across
  a small field), but TDR probe coordinates still vary individually for
  kriging.
- **ERA5-Land soil moisture reference** (from Open-Meteo) is ~9-11 km
  resolution — useful as a coarse sanity-check covariate, not a substitute
  for the drone-derived estimate. It's `null` on the forecast endpoint
  (only available via the archive endpoint, for dates >5 days old).
- Frame extraction interval defaults to 1 frame every 2s; tune `--interval`
  per flight for ~70-80% forward overlap based on your actual flight speed
  and altitude for good ODM reconstruction.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

If pytest errors on cleanup with a `PermissionError` on a stale
`pytest-of-<user>` temp folder (a Windows file-lock quirk, unrelated to the
tests themselves — all tests will have already reported PASSED above it), run
with `--basetemp=.\outputs\pytest_tmp` to sidestep it.

All 18 tests pass without Docker (the ODM/orthomosaic stage is tested with
mocks). A synthetic end-to-end run (no real flight data needed) is in
`data/sample/generate_synthetic.py` — regenerate and run with:

```powershell
.\.venv\Scripts\python.exe data\sample\generate_synthetic.py
.\.venv\Scripts\python.exe -m pipeline.cli run --video data\sample\synthetic_flight.mp4 `
  --tdr-csv data\sample\tdr_readings.csv --site-lat 35.71 --site-lon -78.85 `
  --date 2026-07-25 --out outputs\synthetic_run `
  --orthomosaic data\sample\synthetic_orthomosaic.tif
```
