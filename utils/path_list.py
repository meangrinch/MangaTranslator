from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from utils.logging import log_message

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FAILED_PATHS_FILENAME = "failed_paths.txt"


def resolve_source_path(
    img_path: Union[str, Path],
    source_path_map: Optional[Dict[str, str]] = None,
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


def write_failed_paths(
    output_dir: Union[str, Path],
    paths: Iterable[str],
) -> Optional[Path]:
    """Write unique absolute paths to ``{output_dir}/failed_paths.txt``.

    Returns the file path when something was written, otherwise None.
    Paths are recorded as given (after resolve); callers may pass Gradio
    cache paths or real disk paths depending on how inputs were provided.
    """
    unique: List[str] = []
    seen = set()
    for raw in paths:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            abs_path = str(Path(text).resolve())
        except OSError:
            abs_path = text
        if abs_path in seen:
            continue
        seen.add(abs_path)
        unique.append(abs_path)

    if not unique:
        return None

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / FAILED_PATHS_FILENAME
        out_file.write_text("\n".join(unique) + "\n", encoding="utf-8")
        return out_file
    except OSError as e:
        log_message(
            f"Warning: failed to write {FAILED_PATHS_FILENAME}: {e}",
            always_print=True,
        )
        return None


def read_image_paths_from_txt(txt_path: Union[str, Path]) -> List[Path]:
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

    found: List[Path] = []
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
