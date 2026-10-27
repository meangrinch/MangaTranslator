from pathlib import Path

from core.ml.model_manager import ModelType, get_model_manager, migrate_model_dirs
from core.validation import autodetect_yolo_model_path

OLD_LAYOUT_FILES = {
    "flux/CACHEDIR.TAG": "inpainting/diffusers/CACHEDIR.TAG",
    "flux/models--Disty0--klein/refs/main": "inpainting/diffusers/models--Disty0--klein/refs/main",
    "flux/sdcpp/flux2-vae.safetensors": "inpainting/sdcpp/flux2-vae.safetensors",
    "flux/sdcpp/kontext/ae.safetensors": "inpainting/sdcpp/kontext/ae.safetensors",
    "lama/lama-manga.safetensors": "inpainting/lama/lama-manga.safetensors",
    "yolo/animetext_yolov12x.pt": "detection/yolo/animetext_yolov12x.pt",
    "rtdetr/comic-text-and-bubble-detector/model.safetensors": (
        "detection/rtdetr/comic-text-and-bubble-detector/model.safetensors"
    ),
    "sam/CACHEDIR.TAG": "detection/sam/CACHEDIR.TAG",
    "manga-ocr-base/model.safetensors": "ocr/manga-ocr-base/model.safetensors",
    "paddleocr-vl-1.6/config.json": "ocr/paddleocr-vl-1.6/config.json",
    "sdcpp/master-1015-e41fbcf/sd-server.exe": (
        "runtime/sdcpp/master-1015-e41fbcf/sd-server.exe"
    ),
    "upscale/2x-AnimeSharpV4_RCAN.safetensors": "upscale/2x-AnimeSharpV4_RCAN.safetensors",
}


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_old_model_folders_move_into_the_task_layout(tmp_path):
    for old in OLD_LAYOUT_FILES:
        _write(tmp_path / old, old)

    migrate_model_dirs(tmp_path)

    for old, new in OLD_LAYOUT_FILES.items():
        assert (tmp_path / new).read_text(encoding="utf-8") == old
    for old_dir in ("flux", "lama", "yolo", "rtdetr", "sam", "sdcpp"):
        assert not (tmp_path / old_dir).exists()
    assert not (tmp_path / "manga-ocr-base").exists()


def test_migration_keeps_both_copies_when_the_new_folder_exists(tmp_path):
    _write(tmp_path / "yolo" / "old.pt")
    _write(tmp_path / "detection" / "yolo" / "new.pt")

    migrate_model_dirs(tmp_path)

    assert (tmp_path / "yolo" / "old.pt").exists()
    assert not (tmp_path / "detection" / "yolo" / "old.pt").exists()


def test_migration_is_a_no_op_on_a_fresh_install(tmp_path):
    migrate_model_dirs(tmp_path)
    migrate_model_dirs(tmp_path / "missing")

    assert list(tmp_path.iterdir()) == []


def test_model_paths_follow_the_task_layout():
    manager = get_model_manager()
    root = manager.models_dir
    paths = manager.model_paths

    assert paths[ModelType.YOLO_OSBTEXT].parent == root / "detection" / "yolo"
    assert (
        paths[ModelType.RTDETR_CONJOINED_BUBBLE].parent == root / "detection" / "rtdetr"
    )
    assert paths[ModelType.MANGA_OCR].parent == root / "ocr"
    assert paths[ModelType.PADDLE_OCR_VL].parent == root / "ocr"
    assert paths[ModelType.LAMA_MANGA].parent == root / "inpainting" / "lama"
    assert paths[ModelType.TEXT_SAM].parent == root / "detection" / "text_sam"
    assert paths[ModelType.FLUX2_SDCPP_VAE].parent == root / "inpainting" / "sdcpp"
    assert paths[ModelType.SDCPP_SERVER].parent == root / "runtime" / "sdcpp"
    assert paths[ModelType.UPSCALE].parent == root / "upscale"
    assert manager.diffusers_cache_dir == root / "inpainting" / "diffusers"
    assert manager.sam_cache_dir == root / "detection" / "sam"


def test_a_custom_models_dir_moves_every_download(tmp_path):
    manager = get_model_manager()
    original = manager.models_dir
    try:
        manager.set_models_dir(tmp_path)

        assert manager.models_dir == tmp_path.resolve()
        for path in manager.model_paths.values():
            assert path.is_relative_to(tmp_path.resolve())
        assert manager.diffusers_cache_dir.is_relative_to(tmp_path.resolve())
        assert manager.sam_cache_dir.is_relative_to(tmp_path.resolve())
        assert manager._flux_sdcpp_base_dir().is_relative_to(tmp_path.resolve())
        assert (
            manager.sdcpp_server_manager.install_dir
            == manager.model_paths[ModelType.SDCPP_SERVER]
        )
        path = autodetect_yolo_model_path(tmp_path, "yolo_2")
        assert path == manager.model_paths[ModelType.YOLO_SPEECH_BUBBLE_2]
    finally:
        manager.set_models_dir(original)


def test_validated_yolo_path_matches_the_managers_path():
    manager = get_model_manager()

    for detector, model_type in (
        ("yolo_1", ModelType.YOLO_SPEECH_BUBBLE),
        ("yolo_2", ModelType.YOLO_SPEECH_BUBBLE_2),
    ):
        path = autodetect_yolo_model_path(manager.models_dir, detector)
        assert path == manager.model_paths[model_type]
