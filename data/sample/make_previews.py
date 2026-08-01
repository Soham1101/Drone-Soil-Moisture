"""Generate PNG previews of pipeline outputs for quick visual inspection."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

OUT = Path(__file__).resolve().parent.parent.parent / "outputs" / "previews"
OUT.mkdir(parents=True, exist_ok=True)


def preview_orthophoto(path, out_name, max_size=900):
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])
    img = Image.fromarray(np.moveaxis(arr, 0, -1))
    img.thumbnail((max_size, max_size))
    img.save(OUT / out_name)


def preview_vwc_map(path, out_name, max_size=900):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata

    valid = arr[arr != nodata]
    lo, hi = valid.min(), valid.max()
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)

    # Simple dry (tan) -> wet (blue) colormap.
    dry = np.array([214, 174, 116])
    wet = np.array([30, 90, 180])
    rgb = (dry[None, None, :] * (1 - norm[..., None]) + wet[None, None, :] * norm[..., None]).astype("uint8")
    rgb[arr == nodata] = [40, 40, 40]

    img = Image.fromarray(rgb)
    img = img.resize((max_size, max_size), Image.NEAREST)
    img.save(OUT / out_name)


if __name__ == "__main__":
    preview_orthophoto(
        "outputs/brighton_beach_odm/odm_orthophoto/odm_orthophoto.tif",
        "brighton_beach_orthophoto.png",
    )
    preview_vwc_map(
        "outputs/synthetic_run/vwc_prediction_map.tif",
        "synthetic_vwc_map.png",
    )
    print("Previews written to", OUT)
