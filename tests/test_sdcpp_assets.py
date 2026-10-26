from pathlib import Path

import pytest

from core.ml.model_manager import get_model_manager
from core.ml.sdcpp_server import SDCppServerManager
from utils.exceptions import ModelError
from utils.model_metadata import FLUX_SDCPP_INT8_CONVROT


@pytest.fixture
def downloads(monkeypatch):
    """Resolve sd.cpp assets without downloading; returns the (repo, filename, target) calls."""
    manager = get_model_manager()
    calls = []

    def fake_ensure_hf_file(repo_id, filename, target, token=None, verbose=False):
        calls.append((repo_id, filename, Path(target)))
        return Path(target)

    monkeypatch.setattr(manager, "_ensure_hf_file", fake_ensure_hf_file)
    monkeypatch.setattr(
        manager.sdcpp_server_manager,
        "ensure_server_executable",
        lambda verbose=False: Path("sd-server.exe"),
    )
    return manager, calls


def _server_flags(assets):
    args = assets["server_args"]
    return dict(zip(args[::2], (assets[key] for key in args[1::2]), strict=True))


def test_fibo_gguf_text_encoder_brings_its_vision_projector(downloads):
    manager, calls = downloads

    assets = manager.ensure_flux_sdcpp_assets(
        "fibo2_edit_turbo", text_encoder_quant="Q4_K_XL"
    )

    flags = _server_flags(assets)
    assert flags["--llm"].name == "Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf"
    assert flags["--llm_vision"].name == "Qwen3-VL-4B-Instruct-mmproj-F16.gguf"
    assert ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf") in [
        call[:2] for call in calls
    ]
    assert assets["diffusion_model"].name == (
        "fibo2_edit_turbo_int8_convrot.safetensors"
    )


def test_int8_text_encoder_needs_no_vision_projector(downloads):
    manager, calls = downloads

    assets = manager.ensure_flux_sdcpp_assets(
        "qwen_image_2_1_turbo",
        diffusion_quant=FLUX_SDCPP_INT8_CONVROT,
        text_encoder_quant=FLUX_SDCPP_INT8_CONVROT,
    )

    assert "--llm_vision" not in assets["server_args"]
    repos = {call[1]: call[0] for call in calls}
    assert (
        repos["diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"]
        == "Comfy-Org/Qwen-Image-2.1"
    )
    assert (
        repos["Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"]
        == "unsloth/Qwen-Image-2.1-FP8"
    )
    assert assets["vae"].name == "qwen_image_2.1_vae_bf16.safetensors"
    # Nested repo paths land flat in the sd.cpp model folder
    assert assets["diffusion_model"].parent == manager._flux_sdcpp_base_dir()


def test_qwen_turbo_uses_its_sigma_schedule(downloads):
    manager, _ = downloads

    assets = manager.ensure_flux_sdcpp_assets("qwen_image_2_1_turbo")

    assert list(assets["extra_args"]) == ["--sigmas", "qwen-image-2.1-turbo"]
    assert _server_flags(assets)["--llm_vision"].name == (
        "Qwen3-VL-8B-Instruct-mmproj-F16.gguf"
    )


def test_flux3_shares_fibo_text_encoder_and_klein_vae(downloads):
    manager, _ = downloads

    flux3 = manager.ensure_flux_sdcpp_assets("flux3_klein", text_encoder_quant="Q8_0")
    fibo = manager.ensure_flux_sdcpp_assets(
        "fibo2_edit_turbo", text_encoder_quant="Q8_0"
    )
    klein = manager.ensure_flux_sdcpp_assets("flux_klein_4b")

    assert flux3["llm"] == fibo["llm"]
    assert flux3["vae"] == fibo["vae"] == klein["vae"]
    # FLUX.3's flux2 reference preset doesn't use the vision encoder
    assert "--llm_vision" not in flux3["server_args"]
    assert list(flux3["extra_args"]) == []


def test_flux3_int8_diffusion_model_comes_from_the_comfy_mirror(downloads):
    manager, calls = downloads

    manager.ensure_flux_sdcpp_assets(
        "flux3_klein", diffusion_quant=FLUX_SDCPP_INT8_CONVROT
    )

    assert (
        "ml-resources/flux-3-image-klein-comfy",
        "diffusion_models/flux3-image-klein-int8-convrot.safetensors",
    ) in [call[:2] for call in calls]


def test_step_skip_runs_sdcpp_step_cache_with_quarter_warmup(tmp_path):
    args, mode, warmup, steps = SDCppServerManager(tmp_path)._cache_args("step-skip", 8)

    assert args == [
        "--cache-mode",
        "dbcache",
        "--cache-option",
        "threshold=0.08,warmup=2",
    ]
    assert (mode, warmup, steps) == ("step-skip", 2, 8)


@pytest.mark.parametrize("mode", ["cache-dit", "taylorseer", "dbcache"])
def test_retired_dit_cache_modes_are_rejected(tmp_path, mode):
    with pytest.raises(ModelError, match="Unknown sd.cpp cache mode"):
        SDCppServerManager(tmp_path)._cache_args(mode, 4)


def test_server_command_passes_model_specific_args(tmp_path):
    server_manager = SDCppServerManager(tmp_path)
    assets = {
        "executable": Path("sd-server.exe"),
        "diffusion_model": Path("dit.gguf"),
        "vae": Path("vae.safetensors"),
        "llm": Path("te.gguf"),
        "llm_vision": Path("mmproj.gguf"),
        "server_args": ("--llm", "llm", "--llm_vision", "llm_vision"),
        "extra_args": ("--sigmas", "qwen-image-2.1-turbo"),
    }

    cmd = server_manager._server_command(
        "qwen_image_2_1_turbo", assets, port=1234, cache_args=[], steps=8
    )

    assert cmd[cmd.index("--flash-attn") + 1] == "on"
    assert "--fa" not in cmd
    assert cmd[cmd.index("--llm_vision") + 1] == "mmproj.gguf"
    assert cmd[cmd.index("--sigmas") + 1] == "qwen-image-2.1-turbo"
    assert cmd[cmd.index("--guidance") + 1] == "1.0"


def _install_current_build(tmp_path, monkeypatch) -> SDCppServerManager:
    sdcpp_dir = tmp_path / "sdcpp"
    server_manager = SDCppServerManager(sdcpp_dir / SDCppServerManager.RELEASE_TAG)
    monkeypatch.setattr(
        server_manager, "_select_archive_name", lambda: "sd-master-new-bin-win.zip"
    )
    build_dir = server_manager.install_dir / "sd-master-new-bin-win"
    build_dir.mkdir(parents=True)
    (build_dir / "sd-server.exe").write_bytes(b"")
    (build_dir / "sd-server").write_bytes(b"")
    return server_manager


def test_outdated_sdcpp_builds_are_removed_once_current_build_is_ready(
    tmp_path, monkeypatch
):
    server_manager = _install_current_build(tmp_path, monkeypatch)
    sdcpp_dir = server_manager.install_dir.parent
    old_build = sdcpp_dir / "master-778-c00a9e9" / "sd-master-c00a9e9-bin-win"
    old_build.mkdir(parents=True)
    (old_build / "sd-server.exe").write_bytes(b"old")
    (sdcpp_dir / "my-custom-build").mkdir()
    (sdcpp_dir / "notes.txt").write_text("keep", encoding="utf-8")

    server_manager.ensure_server_executable()

    assert not (sdcpp_dir / "master-778-c00a9e9").exists()
    assert server_manager.install_dir.exists()
    assert (sdcpp_dir / "my-custom-build").exists()
    assert (sdcpp_dir / "notes.txt").exists()


def test_sdcpp_binary_override_leaves_managed_builds_alone(tmp_path, monkeypatch):
    server_manager = _install_current_build(tmp_path, monkeypatch)
    old_build = server_manager.install_dir.parent / "master-778-c00a9e9"
    old_build.mkdir()
    override = tmp_path / "sd-server.exe"
    override.write_bytes(b"")
    monkeypatch.setenv(SDCppServerManager.SERVER_BINARY_ENV, str(override))

    assert server_manager.ensure_server_executable() == override
    assert old_build.exists()
