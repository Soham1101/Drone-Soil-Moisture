import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
from rasterio.transform import from_origin

from pipeline.problem_areas import DRY_HIGHLIGHT_COLOR, WET_HIGHLIGHT_COLOR
from pipeline.view3d import _vwc_to_color, build_3d_deck


def _write_vwc_raster(path, nodata=-9999.0):
    transform = from_origin(-79.0, 36.0, 0.0005, 0.0005)
    data = np.linspace(15, 35, 16, dtype="float32").reshape(4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_rgb_ortho(path):
    transform = from_origin(-79.0, 36.0, 0.0005, 0.0005)
    data = np.random.default_rng(0).integers(0, 255, (3, 4, 4), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_build_3d_deck_returns_deck_with_expected_layers(tmp_path):
    vwc_path = tmp_path / "vwc.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_vwc_raster(vwc_path)
    _write_rgb_ortho(ortho_path)

    tdr_points = pd.DataFrame({"lat": [36.001, 35.999], "lon": [-78.999, -79.001], "vwc_percent": [20.0, 25.0]})

    deck = build_3d_deck(str(vwc_path), str(ortho_path), tdr_points=tdr_points, grid_size=4)

    assert isinstance(deck, pdk.Deck)
    layer_types = [layer.type for layer in deck.layers]
    assert layer_types == ["BitmapLayer", "ColumnLayer", "ScatterplotLayer"]


def test_build_3d_deck_without_tdr_points(tmp_path):
    vwc_path = tmp_path / "vwc.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_vwc_raster(vwc_path)
    _write_rgb_ortho(ortho_path)

    deck = build_3d_deck(str(vwc_path), str(ortho_path), tdr_points=None, grid_size=4)

    layer_types = [layer.type for layer in deck.layers]
    assert layer_types == ["BitmapLayer", "ColumnLayer"]


def test_build_3d_deck_image_is_remote_url_not_data_uri(tmp_path):
    vwc_path = tmp_path / "vwc.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_vwc_raster(vwc_path)
    _write_rgb_ortho(ortho_path)

    deck = build_3d_deck(str(vwc_path), str(ortho_path), grid_size=4)

    bitmap_layer = deck.layers[0]
    image_value = str(bitmap_layer.image)
    assert image_value.startswith("http://")
    assert "data:image" not in image_value


def test_vwc_to_color_highlights_dry_and_wet_values():
    vwc = pd.Series([10.0, 20.0, 30.0, 40.0])
    colors = _vwc_to_color(vwc, dry_value=15.0, wet_value=35.0)

    assert colors[0] == DRY_HIGHLIGHT_COLOR.tolist()
    assert colors[3] == WET_HIGHLIGHT_COLOR.tolist()
    assert colors[1] not in (DRY_HIGHLIGHT_COLOR.tolist(), WET_HIGHLIGHT_COLOR.tolist())


def test_build_3d_deck_passes_through_highlight_thresholds(tmp_path):
    vwc_path = tmp_path / "vwc.tif"
    ortho_path = tmp_path / "ortho.tif"
    _write_vwc_raster(vwc_path)
    _write_rgb_ortho(ortho_path)

    deck = build_3d_deck(str(vwc_path), str(ortho_path), grid_size=4, dry_value=17.0, wet_value=33.0)

    column_layer = deck.layers[1]
    records = column_layer.data
    assert records[0]["fill_color"] == DRY_HIGHLIGHT_COLOR.tolist()
    assert records[-1]["fill_color"] == WET_HIGHLIGHT_COLOR.tolist()
