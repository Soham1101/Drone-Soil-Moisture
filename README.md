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
and inline results:

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port 8765 --server.enableStaticServing true
```

or double-click `run_app.bat`. It opens at http://localhost:8765. Check "Use
bundled synthetic demo data" to try it immediately with no inputs of your own.

**3D Field View** tab: a deck.gl view with the stitched orthomosaic draped
flat on the ground and a grid of 3D columns extruded above it — column height
*and* color both encode predicted VWC (tan = dry, blue = wet), with TDR probe
locations marked in red. Drag to orbit, scroll to zoom, right-click-drag (or
ctrl-drag) to tilt. Sliders control grid resolution, elevation exaggeration,
tilt, and rotation. **2D Map** and **Weather + Downloads** tabs hold the flat
map previews and file downloads.

`--server.enableStaticServing true` is required — the 3D view publishes the
orthomosaic preview to Streamlit's static file server (`./static/`, gitignored)
rather than embedding it as a data URI, because Streamlit's pydeck bridge
can't parse a `data:image/...;base64,...` string as an image prop. If you
change the port in `run_app.bat`, also update `APP_BASE_URL` in
`pipeline/config.py` to match — the 3D view builds an absolute URL from it.

**Problem-area thresholds**: sliders (in the main Results section) flag cells
in the bottom/top percentile of *this field's own* VWC distribution as dry
(red) or wet (purple), highlighted on both the 3D view and 2D map. These are
relative outliers, not an absolute agronomic standard — tune the percentiles
to taste.

**Crop Health / Germination Monitoring** (bottom of the page): independent of
the soil-moisture model — analyzes a *separate* RGB flight (e.g. DJI Avata,
no thermal camera needed) via Excess Green Index (ExG), a proxy for live
vegetation vs. bare soil, to flag likely non-germinated patches. No true NDVI
is possible without a near-infrared band, so treat this as a proxy, not a
calibrated vegetation index. Accepts a pre-built RGB orthomosaic GeoTIFF or a
raw RGB video (frame-extracted + built via ODM the same way as the thermal
flight). Check "Use bundled synthetic RGB demo data" to try it immediately.

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

All 30 tests pass without Docker (the ODM/orthomosaic stage is tested with
mocks). A synthetic end-to-end run (no real flight data needed) is in
`data/sample/generate_synthetic.py` — regenerate and run with:

```powershell
.\.venv\Scripts\python.exe data\sample\generate_synthetic.py
.\.venv\Scripts\python.exe -m pipeline.cli run --video data\sample\synthetic_flight.mp4 `
  --tdr-csv data\sample\tdr_readings.csv --site-lat 35.71 --site-lon -78.85 `
  --date 2026-07-25 --out outputs\synthetic_run `
  --orthomosaic data\sample\synthetic_orthomosaic.tif
```
