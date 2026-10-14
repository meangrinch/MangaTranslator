"""Terminology memory: a per-series glossary file of `source -> translation` lines."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

NEW_TERMS_MARKER = "NEW TERMS:"

# Only a line that starts with the marker counts, so an item mentioning it is safe
_MARKER_LINE = re.compile(r"(?im)^[ \t#*`>-]*new terms[ \t*`]*:?[ \t*`]*(.*)$")
_READING = re.compile(r"\([^)]*\)|（[^）]*）")
_BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_ARROW = re.compile(r"\s*(?:->|→)\s*")
_lock = threading.RLock()


GLOSSARY_DIR = Path("output") / ".glossary"


def default_glossary_path(name: str) -> Path:
    return GLOSSARY_DIR / f"{name}_{time.strftime('%Y-%m-%d')}.txt"


def normalize_source(text: str) -> str:
    """Drop parenthesized readings (furigana) so `猫猫(マオマオ)` keys as `猫猫`."""
    return _READING.sub("", text).strip()


def _parse_pair(line: str) -> tuple[str, str] | None:
    line = _BULLET.sub("", line.strip().strip("`").strip())
    parts = _ARROW.split(line, maxsplit=1)
    if len(parts) != 2:
        return None
    source, translation = (part.strip().strip("*`").strip() for part in parts)
    return (source, translation) if source and translation else None


def load_terms(path: str | Path) -> dict[str, str]:
    """Read the glossary; the first entry for a source wins."""
    try:
        with _lock:
            text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    terms: dict[str, str] = {}
    for line in text.splitlines():
        pair = _parse_pair(line)
        if pair:
            terms.setdefault(*pair)
    return terms


def format_glossary_section(terms: dict[str, str]) -> str:
    if not terms:
        return ""
    lines = "\n".join(f"{source} -> {tr}" for source, tr in terms.items())
    return f"\n## GLOSSARY\n{lines}\n"


def split_new_terms(
    response_text: str | None,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Split a response into its numbered part and the proposed `NEW TERMS` pairs."""
    match = _MARKER_LINE.search(response_text or "")
    if not match:
        return response_text, []
    head = response_text[: match.start()].rstrip()
    # A fence opened just for the terms block would otherwise join the last item
    while "\n" in head and head.rsplit("\n", 1)[1].strip().startswith("```"):
        head = head.rsplit("\n", 1)[0].rstrip()
    lines = [match.group(1), *response_text[match.end() :].splitlines()]
    return head, [pair for line in lines if (pair := _parse_pair(line))]


def record_new_terms(
    path: str | Path,
    pairs: list[tuple[str, str]],
    transcriptions: list[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Append unseen terms whose source appears on the page; returns (added, rejected)."""
    path = Path(path)
    page_text = "\n".join(normalize_source(text) for text in transcriptions)
    with _lock:
        known = {normalize_source(source) for source in load_terms(path)}
        added: list[tuple[str, str]] = []
        rejected: list[tuple[str, str]] = []
        for raw_source, translation in pairs:
            source = normalize_source(raw_source)
            if not source or source in known:
                continue
            if source not in page_text:
                rejected.append((source, translation))
                continue
            known.add(source)
            added.append((source, translation))

        if added:
            existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            new_lines = "".join(f"{source} -> {tr}\n" for source, tr in added)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + ".tmp")
            try:
                tmp_path.write_text(existing + new_lines, encoding="utf-8")
                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)
    return added, rejected
