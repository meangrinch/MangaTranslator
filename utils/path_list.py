from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from utils.logging import log_message

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FAILED_PATHS_FILENAME = "failed_paths.txt"
FAILED_DIRNAME = "failed"


def resolve_source_path(
    img_path: str | Path,
    source_path_map: dict[str, str] | None = None,
) -> str:
    """Return absolute source path, remapping through source_path_map when present."""
    path = Path(img_path)
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)

    if not source_path_map:
        return resolved

    if resolved in source_path_map:
        return source_path_map[resolved]

    as_str = str(path)
    if as_str in source_path_map:
        return source_path_map[as_str]

    return resolved


def record_failed_images(
    output_dir: str | Path,
    failures: Iterable[tuple[Path, Path]],
    succeeded: Iterable[str | Path],
    glossary_path: str | None = None,
) -> Path | None:
    """Keep copies of failed images in ``{output_dir}/failed`` and list them.

    *failures* pairs each failed image's current path with its path relative
    to the run, which is mirrored under ``failed/``. Copies of *succeeded*
    images left there by an earlier run are deleted. ``failed_paths.txt``
    records the run folder and glossary so a retry writes back into this run.
    Returns the list file when something failed, otherwise None.
    """
    out_dir = Path(output_dir).resolve()
    failed_root = out_dir / FAILED_DIRNAME

    for src in succeeded:
        src = Path(src).resolve()
        if src.is_relative_to(failed_root):
            src.unlink(missing_ok=True)

    copies: list[Path] = []
    for current, rel in failures:
        dest = failed_root / rel
        try:
            if dest.resolve() != Path(current).resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current, dest)
            if dest not in copies:
                copies.append(dest)
        except OSError as e:
            log_message(
                f"Warning: could not keep a copy of failed image {current}: {e}",
                always_print=True,
            )

    if failed_root.is_dir():
        for root, _, _ in os.walk(failed_root, topdown=False):
            try:
                Path(root).rmdir()
            except OSError:
                pass  # not empty

    out_file = out_dir / FAILED_PATHS_FILENAME
    try:
        if not copies:
            out_file.unlink(missing_ok=True)
            return None
        headers = [f"# output_dir: {out_dir}"]
        if glossary_path:
            headers.append(f"# glossary: {Path(glossary_path).resolve()}")
        out_file.write_text(
            "\n".join(headers + [str(p) for p in copies]) + "\n", encoding="utf-8"
        )
        return out_file
    except OSError as e:
        log_message(
            f"Warning: failed to write {FAILED_PATHS_FILENAME}: {e}",
            always_print=True,
        )
        return None


def read_path_list_headers(txt_path: str | Path) -> dict[str, str]:
    """Return ``# key: value`` lines from a path list as a dict."""
    try:
        lines = Path(txt_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    headers = {}
    for line in lines:
        text = line.strip()
        if not text.startswith("#"):
            continue
        key, sep, value = text[1:].partition(":")
        if sep:
            headers[key.strip()] = value.strip()
    return headers


def stage_path_list(
    txt_path: str | Path,
    dest_dir: Path,
) -> tuple[dict[str, str], Path | None, str | None]:
    """Copy a path list's images into *dest_dir* for a batch run.

    Returns (staged→listed path map, run output folder, glossary path). Lists
    written by record_failed_images name their run, so their images are staged
    at their path under that run's ``failed/`` folder and the batch can write
    them back with the original structure; other lists are staged flat.
    """
    headers = read_path_list_headers(txt_path)
    run_dir = Path(headers["output_dir"]) if headers.get("output_dir") else None
    failed_root = (run_dir / FAILED_DIRNAME).resolve() if run_dir else None

    source_path_map: dict[str, str] = {}
    for index, img_file in enumerate(read_image_paths_from_txt(txt_path)):
        rel = Path(img_file.name)
        if failed_root and img_file.is_relative_to(failed_root):
            rel = img_file.relative_to(failed_root)
        dest = dest_dir / rel
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}_{index}{dest.suffix}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, dest)
            source_path_map[str(dest.resolve())] = str(img_file)
        except OSError as e:
            log_message(
                f"Warning: failed to copy {img_file}: {e}",
                always_print=True,
            )
    return source_path_map, run_dir, headers.get("glossary")


def read_image_paths_from_txt(txt_path: str | Path) -> list[Path]:
    """Read a path list file; return existing image files only.

    Skips blank lines, ``#`` comments, missing paths, and non-image extensions.
    """
    path = Path(txt_path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        log_message(
            f"Warning: could not read path list '{path}': {e}",
            always_print=True,
        )
        return []

    found: list[Path] = []
    seen = set()
    for line in content.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        # Allow optional surrounding quotes
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1].strip()
        if not text:
            continue

        candidate = Path(text)
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate

        if not resolved.is_file():
            continue
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        found.append(resolved)

    return found
