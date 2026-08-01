"""Streamlit GUI for the soil moisture pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.frame_extraction import extract_frames
from pipeline.orthomosaic import build_orthomosaic
from pipeline.preview import orthophoto_to_png, vwc_map_to_png
from pipeline.problem_areas import compute_dry_wet_thresholds, problem_area_fractions
from pipeline.runner import run_pipeline
from pipeline.vegetation import vegetation_health_to_png
from pipeline.view3d import build_3d_deck

st.set_page_config(page_title="Soil Moisture Pipeline", layout="wide")
st.title("Thermal UAV Soil Moisture Pipeline")
st.caption(
    "Upload drone footage + TDR probe readings to get a predicted volumetric "
    "water content (VWC) map via regression kriging on land-surface temperature."
)

UPLOADS_DIR = Path("outputs/app_uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VIDEO = Path("data/sample/synthetic_flight.mp4")
SAMPLE_TDR = Path("data/sample/tdr_readings.csv")
SAMPLE_ORTHO = Path("data/sample/synthetic_orthomosaic.tif")
SAMPLE_VEG_RGB = Path("data/sample/synthetic_veg_rgb.tif")


def _save_upload(uploaded_file, subdir: str) -> str:
    target_dir = UPLOADS_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path)


use_demo_data = st.checkbox(
    "Use bundled synthetic demo data instead of uploading files",
    help="Runs the same synthetic dataset used to validate this pipeline, so you can see it work with no inputs of your own.",
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")

    orthomosaic_file = st.file_uploader(
        "Pre-built orthomosaic GeoTIFF (optional — skips frame extraction + OpenDroneMap)",
        type=["tif", "tiff"],
        disabled=use_demo_data,
    )
    video_file = st.file_uploader(
        "Drone video (.mp4)",
        type=["mp4"],
        disabled=use_demo_data or orthomosaic_file is not None,
        help="Required unless a pre-built orthomosaic is supplied above.",
    )
    sensor = st.selectbox("Sensor type", ["thermal", "rgb"], disabled=use_demo_data)
    interval = st.number_input(
        "Frame extraction interval (seconds)", value=2.0, min_value=0.1, disabled=use_demo_data
    )
    gcp_file = st.file_uploader(
        "GCP file (required if footage has no embedded GPS, e.g. O4/Avata DVR footage)",
        type=["txt"],
        disabled=use_demo_data,
    )
    tdr_file = st.file_uploader("TDR readings CSV", type=["csv"], disabled=use_demo_data)

with col2:
    st.subheader("Site + date")
    site_lat = st.number_input("Site latitude", value=35.71, format="%.6f")
    site_lon = st.number_input("Site longitude", value=-78.85, format="%.6f")
    flight_date = st.date_input("Flight date", value=dt.date(2026, 7, 25))
    run_name = st.text_input("Run name", value="run1")

run_clicked = st.button("Run pipeline", type="primary")

if run_clicked:
    if use_demo_data:
        video_path = str(SAMPLE_VIDEO)
        tdr_path = str(SAMPLE_TDR)
        ortho_path = str(SAMPLE_ORTHO)
        gcp_path = None
    else:
        if not tdr_file:
            st.error("A TDR readings CSV is required.")
            st.stop()
        if not video_file and not orthomosaic_file:
            st.error("Provide either a drone video or a pre-built orthomosaic.")
            st.stop()

        tdr_path = _save_upload(tdr_file, "tdr")
        video_path = _save_upload(video_file, "video") if video_file else None
        ortho_path = _save_upload(orthomosaic_file, "ortho") if orthomosaic_file else None
        gcp_path = _save_upload(gcp_file, "gcp") if gcp_file else None

    out_dir = f"outputs/{run_name}"
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)

    status_box = st.status("Running pipeline...", expanded=True)
    try:
        report = run_pipeline(
            tdr_csv=tdr_path,
            site_lat=site_lat,
            site_lon=site_lon,
            date=flight_date.isoformat(),
            out_dir=out_dir,
            video=video_path,
            sensor=sensor,
            interval=interval,
            gcp_file=gcp_path,
            orthomosaic_override=ortho_path,
            progress_cb=status_box.write,
        )
        status_box.update(label="Done", state="complete")
        st.session_state["report"] = report
    except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure to the user
        status_box.update(label="Failed", state="error")
        st.error(f"Pipeline failed: {exc}")

if "report" in st.session_state:
    report = st.session_state["report"]
    st.divider()
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Leave-one-out RMSE", f"{report['rmse']:.3f}")
    m2.metric("Leave-one-out MAE", f"{report['mae']:.3f}")
    m3.metric("Training points", report["n_training_points"])

    tdr_df = pd.read_csv(report["training_table"])

    st.markdown(
        "**Problem-area thresholds** — flag cells unusually dry/wet relative to "
        "this field's own distribution (not an absolute agronomic standard)"
    )
    th1, th2 = st.columns(2)
    dry_percentile = th1.slider("Dry threshold (bottom percentile)", 1, 40, 10, key="dry_percentile")
    wet_percentile = th2.slider("Wet threshold (top percentile)", 60, 99, 90, key="wet_percentile")

    dry_value, wet_value = compute_dry_wet_thresholds(
        report["vwc_prediction_map"], dry_percentile, wet_percentile
    )
    dry_fraction, wet_fraction = problem_area_fractions(report["vwc_prediction_map"], dry_value, wet_value)

    p1, p2 = st.columns(2)
    p1.metric("Flagged dry area", f"{dry_fraction * 100:.1f}%", help=f"VWC <= {dry_value:.1f}%")
    p2.metric("Flagged wet area", f"{wet_fraction * 100:.1f}%", help=f"VWC >= {wet_value:.1f}%")

    highlight_problem_areas = st.checkbox(
        "Highlight flagged areas on the maps below (red = dry, purple = wet)",
        value=True,
        key="highlight_problem_areas",
    )
    highlight_dry = dry_value if highlight_problem_areas else None
    highlight_wet = wet_value if highlight_problem_areas else None

    tab_3d, tab_2d, tab_data = st.tabs(["3D Field View", "2D Map", "Weather + Downloads"])

    with tab_3d:
        st.caption(
            "Drag to orbit, scroll to zoom, right-click-drag (or ctrl-drag) to tilt. "
            "Bar height and color both encode predicted VWC; red points are TDR probe locations."
        )
        c1, c2, c3, c4 = st.columns(4)
        grid_size = c1.slider("Grid resolution", 5, 40, 20, key="grid_size")
        elevation_scale = c2.slider(
            "Elevation exaggeration (m)", 50, 3000, 400, step=50, key="elevation_scale"
        )
        pitch = c3.slider("Tilt", 0, 80, 45, key="pitch")
        bearing = c4.slider("Rotate", 0, 360, 0, key="bearing")
        try:
            deck = build_3d_deck(
                report["vwc_prediction_map"],
                report["orthomosaic"],
                tdr_points=tdr_df[["lat", "lon", "vwc_percent"]],
                grid_size=grid_size,
                elevation_scale=elevation_scale,
                pitch=pitch,
                bearing=bearing,
                dry_value=highlight_dry,
                wet_value=highlight_wet,
            )
            st.pydeck_chart(deck, use_container_width=True, height=600)
        except Exception as exc:  # noqa: BLE001 -- surface any rendering failure to the user
            st.error(f"3D view failed: {exc}")

    with tab_2d:
        left, right = st.columns(2)
        with left:
            st.markdown("**Orthomosaic**")
            try:
                st.image(orthophoto_to_png(report["orthomosaic"]))
            except Exception:
                st.caption("Orthomosaic preview unavailable for this raster.")
        with right:
            st.markdown(
                "**Predicted VWC map** (tan = drier, blue = wetter"
                + (", red = flagged dry, purple = flagged wet)" if highlight_problem_areas else ")")
            )
            st.image(
                vwc_map_to_png(
                    report["vwc_prediction_map"], dry_value=highlight_dry, wet_value=highlight_wet
                )
            )

    with tab_data:
        st.markdown("**Weather at site + date**")
        st.json(report["weather"])

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download VWC map (GeoTIFF)",
                data=Path(report["vwc_prediction_map"]).read_bytes(),
                file_name="vwc_prediction_map.tif",
            )
        with d2:
            st.download_button(
                "Download training table (CSV)",
                data=Path(report["training_table"]).read_bytes(),
                file_name="training_table.csv",
            )
        with d3:
            st.download_button(
                "Download validation report (JSON)",
                data=Path(report["validation_report"]).read_bytes(),
                file_name="validation_report.json",
            )

st.divider()
st.subheader("Crop Health / Germination Monitoring (RGB)")
st.caption(
    "Independent of the soil-moisture model above — analyzes a separate RGB flight "
    "(e.g. DJI Avata, no thermal camera) using Excess Green Index (ExG), a proxy for "
    "live vegetation vs. bare soil, to flag likely non-germinated patches. No true "
    "NDVI is possible without a near-infrared band."
)

use_veg_demo = st.checkbox(
    "Use bundled synthetic RGB demo data",
    key="use_veg_demo",
    help="A synthetic RGB orthomosaic with a deliberate bare patch, so you can see this work with no footage of your own.",
)
veg_file = st.file_uploader(
    "RGB orthomosaic GeoTIFF (pre-built) or RGB drone video (.mp4)",
    type=["tif", "tiff", "mp4"],
    key="veg_upload",
    disabled=use_veg_demo,
)
bare_percentile = st.slider(
    "Bare-soil sensitivity (bottom percentile of this field's own ExG flagged as bare)",
    5,
    60,
    30,
    key="bare_percentile",
)
analyze_clicked = st.button("Analyze crop health")

if analyze_clicked:
    if use_veg_demo:
        veg_source_path = str(SAMPLE_VEG_RGB)
    else:
        if not veg_file:
            st.error("Upload an RGB orthomosaic or RGB video first.")
            st.stop()
        veg_dir = Path("outputs/veg_uploads")
        veg_dir.mkdir(parents=True, exist_ok=True)
        veg_upload_path = veg_dir / veg_file.name
        veg_upload_path.write_bytes(veg_file.getbuffer())
        veg_source_path = str(veg_upload_path)

    if veg_source_path.lower().endswith(".mp4"):
        veg_status_box = st.status("Building orthomosaic from RGB video...", expanded=True)
        try:
            veg_dir = Path("outputs/veg_uploads")
            frames_dir = veg_dir / "frames"
            veg_status_box.write("Extracting frames...")
            extract_frames(veg_source_path, str(frames_dir), interval_s=2.0, sensor_type="rgb")
            veg_status_box.write("Building orthomosaic with OpenDroneMap (this can take several minutes)...")
            odm_dir = veg_dir / "odm"
            veg_ortho_path = build_orthomosaic(str(frames_dir), str(odm_dir))
            veg_status_box.update(label="Done", state="complete")
        except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure to the user
            veg_status_box.update(label="Failed", state="error")
            st.error(f"Failed to build orthomosaic from video: {exc}")
            st.stop()
    else:
        veg_ortho_path = veg_source_path

    try:
        veg_png_bytes, bare_fraction = vegetation_health_to_png(
            veg_ortho_path, threshold_percentile=bare_percentile
        )
        st.session_state["veg_report"] = {
            "png_bytes": veg_png_bytes,
            "bare_fraction": bare_fraction,
            "ortho_path": veg_ortho_path,
        }
    except Exception as exc:  # noqa: BLE001 -- surface any analysis failure to the user
        st.error(f"Crop health analysis failed: {exc}")

if "veg_report" in st.session_state:
    veg_report = st.session_state["veg_report"]
    st.metric("Flagged bare / non-germinated area", f"{veg_report['bare_fraction'] * 100:.1f}%")
    st.image(
        veg_report["png_bytes"],
        caption="Green = healthy canopy, tan = sparse, red = flagged bare/non-germinated",
    )
