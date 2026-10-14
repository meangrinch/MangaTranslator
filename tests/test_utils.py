import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
from PIL import Image

from core import pipeline
from core.config import MangaTranslatorConfig
from core.scaling import scale_kernel
from utils import update_checker
from utils.archive import copy_comic_info, safe_extract_zip
from utils.path_list import (
    read_image_paths_from_txt,
    record_failed_images,
    stage_path_list,
)


def _zip(tmp_path, names):
    archive = tmp_path / "pages.zip"
    with ZipFile(archive, "w") as zf:
        for name in names:
            zf.writestr(name, b"data")
    return archive


def test_safe_extract_zip_extracts_nested_files(tmp_path):
    archive = _zip(tmp_path, ["ch1/001.png", "002.png"])
    dest = tmp_path / "out"
    with ZipFile(archive) as zf:
        safe_extract_zip(zf, dest)
    assert (dest / "ch1" / "001.png").read_bytes() == b"data"
    assert (dest / "002.png").exists()


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = _zip(tmp_path, ["ok.png", "../evil.png"])
    dest = tmp_path / "out"
    with ZipFile(archive) as zf, pytest.raises(ValueError, match="unsafe path"):
        safe_extract_zip(zf, dest)
    assert not (tmp_path / "evil.png").exists()
    assert not (dest / "ok.png").exists()


def test_copy_comic_info_sets_target_language(tmp_path):
    src = tmp_path / "in"
    (src / "ch2").mkdir(parents=True)
    (src / "ComicInfo.xml").write_text(
        "<ComicInfo><Series>S</Series><LanguageISO>ja</LanguageISO></ComicInfo>",
        encoding="utf-8",
    )
    (src / "ch2" / "comicinfo.xml").write_text(
        "<ComicInfo><Series>S2</Series></ComicInfo>", encoding="utf-8"
    )
    out = tmp_path / "out"

    copy_comic_info(src, out, "chinese (simplified)", preserve_structure=True)

    root = ET.parse(out / "ComicInfo.xml").getroot()
    assert root.findtext("Series") == "S"
    assert root.findtext("LanguageISO") == "zh-Hans"
    assert (
        ET.parse(out / "ch2" / "comicinfo.xml").getroot().findtext("LanguageISO")
        == "zh-Hans"
    )


def test_copy_comic_info_drops_unknown_language_and_skips_subdirs(tmp_path):
    src = tmp_path / "in"
    (src / "ch2").mkdir(parents=True)
    xml = "<ComicInfo><LanguageISO>ja</LanguageISO></ComicInfo>"
    (src / "ComicInfo.xml").write_text(xml, encoding="utf-8")
    (src / "ch2" / "ComicInfo.xml").write_text(xml, encoding="utf-8")
    out = tmp_path / "out"

    copy_comic_info(src, out, "Klingon", preserve_structure=False)

    assert ET.parse(out / "ComicInfo.xml").getroot().find("LanguageISO") is None
    assert not (out / "ch2").exists()


@pytest.mark.parametrize(
    ("current", "latest", "expected"),
    [
        ("1.2.3", "v1.2.4", True),
        ("v1.9.0", "1.10.0", True),
        ("1.2.3", "1.2.3", False),
        ("2.0.0", "v1.9.9", False),
    ],
)
def test_is_update_available(current, latest, expected):
    assert update_checker.is_update_available(current, latest) is expected


def test_check_for_update_handles_bad_or_missing_tags(monkeypatch):
    monkeypatch.setattr(update_checker, "get_latest_release_tag", lambda *a: None)
    assert update_checker.check_for_update("1.0.0") == (False, None)

    monkeypatch.setattr(
        update_checker, "get_latest_release_tag", lambda *a: "not-a-version"
    )
    assert update_checker.check_for_update("1.0.0") == (False, None)

    monkeypatch.setattr(update_checker, "get_latest_release_tag", lambda *a: "v2.0.0")
    assert update_checker.check_for_update("1.0.0") == (True, "v2.0.0")


def test_last_seen_version_round_trip_preserves_other_keys(tmp_path):
    config = tmp_path / "config.json"
    assert update_checker.get_last_seen_version(config) is None

    config.write_text('{"theme": "dark"}', encoding="utf-8")
    update_checker.set_last_seen_version(" 1.2.3 ", config)
    assert update_checker.get_last_seen_version(config) == "1.2.3"
    assert '"theme": "dark"' in config.read_text(encoding="utf-8")


def test_read_image_paths_from_txt_keeps_existing_images(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    page = pages / "001.png"
    page.write_bytes(b"")
    note = pages / "notes.md"
    note.write_bytes(b"")

    path_list = tmp_path / "failed_paths.txt"
    path_list.write_text(
        f'# comment\n\n"{page}"\n{note}\n{pages / "missing.png"}\n{page}\n',
        encoding="utf-8",
    )
    assert read_image_paths_from_txt(path_list) == [page.resolve()]


def test_record_failed_images_nothing_to_write(tmp_path):
    assert record_failed_images(tmp_path, [], []) is None
    assert not (tmp_path / "failed_paths.txt").exists()


def test_failed_images_retry_back_into_original_run(monkeypatch, tmp_path):
    # Default glossaries go under ./output/.glossary, so keep them inside tmp_path
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "extracted"
    for rel in ("ch1/001.png", "ch2/001.png", "ch2/002.png"):
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(source / rel)

    failing = {"ch1/001.png", "ch2/001.png"}
    written = []
    glossaries = []

    def fake_translate_and_render(image_path, config, output_path, **_):
        glossaries.append(Path(config.translation.terminology_memory_path).resolve())
        if f"{Path(image_path).parent.name}/{Path(image_path).name}" in failing:
            raise RuntimeError("boom")
        written.append(Path(output_path).resolve().relative_to(run).as_posix())

    monkeypatch.setattr(pipeline, "translate_and_render", fake_translate_and_render)
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )
    config.translation = replace(config.translation, terminology_memory=True)
    run = (tmp_path / "run").resolve()

    results = pipeline.batch_translate_images(
        source, config, run, preserve_structure=True, glossary_name="series"
    )
    glossary = Path(results["glossary_path"]).resolve()
    # A ZIP run's extraction folder is deleted once the batch returns.
    shutil.rmtree(source)

    def retry(path_list):
        staged = Path(tempfile.mkdtemp(dir=tmp_path))
        source_path_map, run_dir, glossary_path = stage_path_list(path_list, staged)
        assert run_dir == run
        assert Path(glossary_path) == glossary
        written.clear()
        return pipeline.batch_translate_images(
            staged,
            config,
            run_dir,
            preserve_structure=True,
            source_path_map=source_path_map,
            glossary_path=glossary_path,
        )

    failing = {"ch2/001.png"}
    results = retry(results["failed_paths_file"])
    assert written == ["ch1/001_translated.png"]
    assert not (run / "failed" / "ch1").exists()
    assert read_image_paths_from_txt(results["failed_paths_file"]) == [
        run / "failed" / "ch2" / "001.png"
    ]

    failing = set()
    results = retry(results["failed_paths_file"])
    assert written == ["ch2/001_translated.png"]
    assert "failed_paths_file" not in results
    assert not (run / "failed").exists()
    assert not (run / "failed_paths.txt").exists()
    assert set(glossaries) == {glossary}


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_page_with_an_empty_bubble_counts_as_failed(
    monkeypatch, tmp_path, parallel_requests
):
    source = tmp_path / "pages"
    source.mkdir()
    for name in ("001.png", "002.png", "003.png"):
        Image.new("RGB", (8, 8), "white").save(source / name)

    def fake_translate_and_render(image_path, config, output_path, **kwargs):
        if Path(image_path).name == "002.png":
            kwargs.get("empty_bubbles_out", []).append((0, 0, 4, 4))

    monkeypatch.setattr(pipeline, "translate_and_render", fake_translate_and_render)
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt",
        device=torch.device("cpu"),
        parallel_requests=parallel_requests,
    )
    config.translation = replace(config.translation, terminology_memory=False)

    results = pipeline.batch_translate_images(source, config, tmp_path / "run")
    assert results["success_count"] == 2
    assert list(results["errors"]) == ["002.png"]
    assert read_image_paths_from_txt(results["failed_paths_file"]) == [
        (tmp_path / "run" / "failed" / "002.png").resolve()
    ]


@pytest.mark.parametrize(
    ("kernel", "scale", "expected"),
    [
        ((3, 3), 1.0, (3, 3)),
        ((3, 5), 2.0, (7, 11)),
        ((31, 31), 4.0, (63, 63)),
        ((3, 3), 0.1, (1, 1)),
    ],
)
def test_scale_kernel_stays_odd_and_bounded(kernel, scale, expected):
    assert scale_kernel(kernel, scale) == expected
