from zipfile import ZipFile

import pytest

from core.scaling import scale_kernel
from utils import update_checker
from utils.archive import safe_extract_zip
from utils.path_list import read_image_paths_from_txt, write_failed_paths


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


def test_failed_paths_round_trip(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    page = pages / "001.png"
    page.write_bytes(b"")
    note = pages / "notes.md"
    note.write_bytes(b"")

    out = write_failed_paths(tmp_path / "out", [str(page), str(page), "", None])
    assert out is not None
    assert out.read_text(encoding="utf-8").splitlines() == [str(page.resolve())]

    out.write_text(
        f'# comment\n\n"{page}"\n{note}\n{pages / "missing.png"}\n{page}\n',
        encoding="utf-8",
    )
    assert read_image_paths_from_txt(out) == [page.resolve()]


def test_write_failed_paths_nothing_to_write(tmp_path):
    assert write_failed_paths(tmp_path, ["", None]) is None
    assert not (tmp_path / "failed_paths.txt").exists()


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
