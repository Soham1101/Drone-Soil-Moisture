"""Extract overlapping still frames from FPV DVR video (DJI O4 Air Unit / Avata).

Neither source embeds GPS/EXIF like a consumer mapping drone, so frames are
plain images -- georeferencing happens later via GCPs in the orthomosaic step.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

VALID_SENSOR_TYPES = {"thermal", "rgb"}


def extract_frames(
    video_path: str,
    out_dir: str,
    interval_s: float = 2.0,
    sensor_type: str = "thermal",
    image_format: str = "png",
) -> list[str]:
    """Extract one frame every `interval_s` seconds of video into `out_dir`.

    `sensor_type` is informational (thermal footage from a non-radiometric FPV
    camera yields a relative thermal proxy, not calibrated absolute temperature --
    consistent with the questionnaire's framing of thermal anomalies as proxies).
    """
    if sensor_type not in VALID_SENSOR_TYPES:
        raise ValueError(f"sensor_type must be one of {VALID_SENSOR_TYPES}, got {sensor_type!r}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(video_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pattern = out / f"frame_%04d.{image_format}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps=1/{interval_s}",
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr}")

    return sorted(str(p) for p in out.glob(f"frame_*.{image_format}"))
