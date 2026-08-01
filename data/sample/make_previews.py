"""Generate PNG previews of pipeline outputs for quick visual inspection."""

from pathlib import Path

from pipeline.preview import orthophoto_to_png, vwc_map_to_png

OUT = Path(__file__).resolve().parent.parent.parent / "outputs" / "previews"
OUT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    (OUT / "brighton_beach_orthophoto.png").write_bytes(
        orthophoto_to_png("outputs/brighton_beach_odm/odm_orthophoto/odm_orthophoto.tif")
    )
    (OUT / "synthetic_vwc_map.png").write_bytes(
        vwc_map_to_png("outputs/synthetic_run/vwc_prediction_map.tif")
    )
    print("Previews written to", OUT)
