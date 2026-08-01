"""Build a georeferenced orthomosaic from extracted stills via OpenDroneMap (Docker).

Frames have no embedded GPS (see frame_extraction.py), so ODM is run in
feature-matching-only mode and georeferenced afterward using a small
ground-control-point (GCP) file -- a handful of known real-world lat/lon/alt
points identified visually (e.g. in QGIS) at features visible in the imagery,
such as field corners.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pipeline.config import ODM_DOCKER_IMAGE


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "version"], capture_output=True, text=True)
    return result.returncode == 0


def build_orthomosaic(images_dir: str, out_dir: str, gcp_file: str | None = None) -> str:
    """Run OpenDroneMap over `images_dir`, writing results under `out_dir`.

    Returns the path to the generated orthophoto GeoTIFF
    (`<out_dir>/odm_orthophoto/odm_orthophoto.tif`).
    """
    if not docker_available():
        raise RuntimeError(
            "Docker is not available (`docker version` failed). "
            "Install Docker Desktop and ensure it is running before calling build_orthomosaic()."
        )

    images_path = Path(images_dir).resolve()
    if not images_path.is_dir() or not any(images_path.iterdir()):
        raise FileNotFoundError(f"No images found in {images_dir}")

    project_dir = Path(out_dir).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    project_images = project_dir / "images"
    project_images.mkdir(exist_ok=True)
    for img in images_path.iterdir():
        target = project_images / img.name
        if not target.exists():
            shutil.copy(img, target)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_dir}:/datasets/project",
        ODM_DOCKER_IMAGE,
        "--project-path",
        "/datasets",
        "project",
    ]

    if gcp_file:
        gcp_path = Path(gcp_file).resolve()
        shutil.copy(gcp_path, project_dir / "gcp_list.txt")
        cmd += ["--gcp", "/datasets/project/gcp_list.txt"]
    # Without a GCP file, ODM produces a relative (non-georeferenced) mosaic.

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OpenDroneMap run failed:\n{result.stderr}")

    orthophoto = project_dir / "odm_orthophoto" / "odm_orthophoto.tif"
    if not orthophoto.is_file():
        raise RuntimeError(
            f"ODM finished but expected output not found at {orthophoto}.\nODM stdout:\n{result.stdout}"
        )

    return str(orthophoto)
