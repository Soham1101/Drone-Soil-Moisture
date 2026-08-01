import subprocess

import pytest

from pipeline.frame_extraction import extract_frames


def _make_test_video(path, duration_s=4, fps=10):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size=64x64:rate={fps}",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_extract_frames_produces_expected_count(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _make_test_video(video_path, duration_s=4)

    out_dir = tmp_path / "frames"
    frames = extract_frames(str(video_path), str(out_dir), interval_s=1.0)

    assert len(frames) == 4
    for f in frames:
        assert f.endswith(".png")


def test_extract_frames_rejects_invalid_sensor_type(tmp_path):
    video_path = tmp_path / "flight.mp4"
    _make_test_video(video_path, duration_s=1)

    with pytest.raises(ValueError):
        extract_frames(str(video_path), str(tmp_path / "frames"), sensor_type="infrared")


def test_extract_frames_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(str(tmp_path / "missing.mp4"), str(tmp_path / "frames"))
