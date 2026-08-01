import pytest

from pipeline import orthomosaic


def test_build_orthomosaic_raises_when_docker_unavailable(tmp_path, mocker):
    mocker.patch.object(orthomosaic, "docker_available", return_value=False)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_0001.png").write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="Docker is not available"):
        orthomosaic.build_orthomosaic(str(images_dir), str(tmp_path / "out"))


def test_build_orthomosaic_raises_when_no_images(tmp_path, mocker):
    mocker.patch.object(orthomosaic, "docker_available", return_value=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        orthomosaic.build_orthomosaic(str(images_dir), str(tmp_path / "out"))


def test_build_orthomosaic_invokes_docker_with_gcp(tmp_path, mocker):
    mocker.patch.object(orthomosaic, "docker_available", return_value=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_0001.png").write_bytes(b"fake")

    gcp_file = tmp_path / "gcp_list.txt"
    gcp_file.write_text("EPSG:4326\n")

    out_dir = tmp_path / "out"
    orthophoto_dir = out_dir / "odm_orthophoto"

    def fake_run(cmd, capture_output, text):
        orthophoto_dir.mkdir(parents=True, exist_ok=True)
        (orthophoto_dir / "odm_orthophoto.tif").write_bytes(b"fake tif")
        result = mocker.Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    run_mock = mocker.patch.object(orthomosaic.subprocess, "run", side_effect=fake_run)

    result_path = orthomosaic.build_orthomosaic(str(images_dir), str(out_dir), gcp_file=str(gcp_file))

    assert result_path == str(orthophoto_dir / "odm_orthophoto.tif")
    called_cmd = run_mock.call_args[0][0]
    assert "--gcp" in called_cmd
    assert (out_dir / "gcp_list.txt").is_file()
    assert (out_dir / "images" / "frame_0001.png").is_file()
