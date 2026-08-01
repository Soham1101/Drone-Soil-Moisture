"""3D deck.gl visualization: VWC bars extruded over the draped orthomosaic.

Renders the stitched aerial imagery as a georeferenced image layer, with a
grid of 3D columns on top -- height and color both encode predicted VWC --
plus the TDR probe locations used as ground truth.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio

from pipeline.config import APP_BASE_URL, PROJECT_ROOT
from pipeline.preview import DRY_COLOR, WET_COLOR, orthophoto_to_png
from pipeline.problem_areas import DRY_HIGHLIGHT_COLOR, WET_HIGHLIGHT_COLOR

# deck.gl's BitmapLayer needs a real URL, not a data: URI -- Streamlit's pydeck
# JSON bridge tries to parse every string prop as an expression and chokes on
# the colons/slashes in a data URI. Publishing to Streamlit's static file
# server (requires --server.enableStaticServing true) sidesteps that.
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)


def _publish_static_image(raster_path: str) -> str:
    png_bytes = orthophoto_to_png(raster_path, max_size=2000)
    digest = hashlib.md5(png_bytes).hexdigest()[:12]
    filename = f"ortho_{digest}.png"
    (STATIC_DIR / filename).write_bytes(png_bytes)
    return f"{APP_BASE_URL}/app/static/{filename}"


def _raster_bounds_lonlat(raster_path: str) -> tuple[float, float, float, float]:
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *bounds)
        else:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    return west, south, east, north


def _sample_grid(vwc_raster_path: str, grid_size: int) -> pd.DataFrame:
    with rasterio.open(vwc_raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    rows = np.linspace(0, data.shape[0] - 1, grid_size).astype(int)
    cols = np.linspace(0, data.shape[1] - 1, grid_size).astype(int)

    records = []
    for r in rows:
        for c in cols:
            value = data[r, c]
            if nodata is not None and value == nodata:
                continue
            x, y = transform * (c + 0.5, r + 0.5)
            records.append({"x": x, "y": y, "vwc": float(value)})

    df = pd.DataFrame(records)
    if crs and crs.to_epsg() != 4326:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(df["x"].to_numpy(), df["y"].to_numpy())
        df["lon"], df["lat"] = lon, lat
    else:
        df["lon"], df["lat"] = df["x"], df["y"]

    return df


def _vwc_to_color(
    vwc: pd.Series, dry_value: float | None = None, wet_value: float | None = None
) -> list:
    lo, hi = vwc.min(), vwc.max()
    span = hi - lo if hi > lo else 1.0
    norm = ((vwc - lo) / span).to_numpy()
    colors = (DRY_COLOR[None, :] * (1 - norm[:, None]) + WET_COLOR[None, :] * norm[:, None]).astype(int)

    vwc_arr = vwc.to_numpy()
    if dry_value is not None:
        colors[vwc_arr <= dry_value] = DRY_HIGHLIGHT_COLOR
    if wet_value is not None:
        colors[vwc_arr >= wet_value] = WET_HIGHLIGHT_COLOR

    return colors.tolist()


def build_3d_deck(
    vwc_raster_path: str,
    orthomosaic_path: str,
    tdr_points: pd.DataFrame | None = None,
    grid_size: int = 20,
    elevation_scale: float = 4000.0,
    pitch: float = 45.0,
    bearing: float = 0.0,
    dry_value: float | None = None,
    wet_value: float | None = None,
) -> pdk.Deck:
    grid_df = _sample_grid(vwc_raster_path, grid_size)
    grid_df["fill_color"] = _vwc_to_color(grid_df["vwc"], dry_value, wet_value)
    grid_df["elevation"] = (grid_df["vwc"] - grid_df["vwc"].min()) / max(
        grid_df["vwc"].max() - grid_df["vwc"].min(), 1e-6
    ) * elevation_scale

    west, south, east, north = _raster_bounds_lonlat(orthomosaic_path)
    bitmap_layer = pdk.Layer(
        "BitmapLayer",
        data=None,
        image=_publish_static_image(orthomosaic_path),
        bounds=[west, south, east, north],
    )

    cell_size_deg = min((east - west) / grid_size, (north - south) / grid_size)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=grid_df,
        get_position="[lon, lat]",
        get_elevation="elevation",
        elevation_scale=1,
        radius=cell_size_deg * 111_000 / 2 * 0.85,
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )

    layers = [bitmap_layer, column_layer]

    if tdr_points is not None and len(tdr_points):
        probe_layer = pdk.Layer(
            "ScatterplotLayer",
            data=tdr_points,
            get_position="[lon, lat]",
            get_radius=cell_size_deg * 111_000 * 0.15,
            get_fill_color=[255, 0, 0],
            pickable=True,
        )
        layers.append(probe_layer)

    view_state = pdk.ViewState(
        latitude=(south + north) / 2,
        longitude=(west + east) / 2,
        zoom=17,
        pitch=pitch,
        bearing=bearing,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider=None,
        map_style=None,
        tooltip={"html": "VWC {vwc}%", "style": {"backgroundColor": "white", "color": "black"}},
    )
