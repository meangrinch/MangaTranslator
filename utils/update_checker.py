from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import Version

from utils.logging import log_message

API_URL = "https://api.github.com/repos/{repo}/releases/latest"
RELEASE_TAG_URL = "https://api.github.com/repos/{repo}/releases/tags/{tag}"


def _resolve_config_file(config_file: Path | None = None) -> Path:
    if config_file is not None:
        return config_file
    from ui.settings_manager import CONFIG_FILE

    return CONFIG_FILE


def get_latest_release_tag(repo: str, timeout: float = 3.0) -> str | None:
    """Return the latest stable release tag from GitHub or None on failure.

    Uses the releases/latest endpoint which excludes drafts and prereleases.
    """
    request = Request(
        API_URL.format(repo=repo),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "MangaTranslator-Updater",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            tag_name = data.get("tag_name")
            return tag_name if isinstance(tag_name, str) else None
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def get_release_notes(
    tag: str, repo: str = "meangrinch/MangaTranslator", timeout: float = 3.0
) -> str | None:
    """Fetch release notes (body) for a specific tag from GitHub or None on failure."""
    request = Request(
        RELEASE_TAG_URL.format(repo=repo, tag=tag),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "MangaTranslator-Updater",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            body = data.get("body")
            return body if isinstance(body, str) else None
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, Exception):
        return None


def get_last_seen_version(config_file: Path | None = None) -> str | None:
    """Retrieve last_seen_version from config.json, or None if not set."""
    target = _resolve_config_file(config_file)
    if not target.exists():
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            version = data.get("last_seen_version")
            return str(version).strip() if version else None
    except Exception:
        return None


def set_last_seen_version(version: str, config_file: Path | None = None) -> None:
    """Save last_seen_version to config.json, creating or updating the file."""
    target = _resolve_config_file(config_file)
    data = {}
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["last_seen_version"] = version.strip()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def format_release_notes(
    version: str, notes: str | None, repo: str = "meangrinch/MangaTranslator"
) -> str:
    """Format release notes with lightweight console separators."""
    clean_ver = normalize_version(version)
    header_title = f" Release Notes (v{clean_ver}) "
    header = f"{header_title:-^80}"
    footer = "-" * 80

    if notes and notes.strip():
        content = notes.strip()
    else:
        content = (
            "Could not fetch changelog from GitHub. View release notes online:\n"
            f"https://github.com/{repo}/releases/tag/v{clean_ver}"
        )

    return f"{header}\n{content}\n{footer}"


def check_and_display_changelog(
    current_version: str,
    repo: str = "meangrinch/MangaTranslator",
    timeout: float = 3.0,
    config_file: Path | None = None,
) -> bool:
    """Check if the app has been updated, and display release notes once in console.

    On fresh install (no prior recorded version), silently records the version.
    Returns True if changelog was displayed, False otherwise.
    """
    last_version = get_last_seen_version(config_file)
    norm_current = normalize_version(current_version)

    if last_version is None:
        set_last_seen_version(norm_current, config_file)
        return False

    if is_update_available(last_version, norm_current):
        tag = f"v{norm_current}"
        notes = get_release_notes(tag, repo=repo, timeout=timeout)
        log_message(
            f"\n{format_release_notes(norm_current, notes, repo=repo)}\n",
            always_print=True,
        )
        set_last_seen_version(norm_current, config_file)
        return True

    return False


def normalize_version(tag: str) -> str:
    """Normalize a tag like 'v1.2.3' to '1.2.3'."""
    return tag.lstrip().lstrip("v").strip()


def is_update_available(current: str, latest: str) -> bool:
    """Return True if latest version is greater than current version."""
    return Version(normalize_version(latest)) > Version(normalize_version(current))


def check_for_update(
    current_version: str,
    repo: str = "meangrinch/MangaTranslator",
    timeout: float = 3.0,
) -> tuple[bool, str | None]:
    """Check GitHub for a newer stable release.

    Returns (True, latest_tag) if newer exists, otherwise (False, None).
    Any failures are treated as no update available.
    """
    latest = get_latest_release_tag(repo, timeout)
    if not latest:
        return False, None
    try:
        return (is_update_available(current_version, latest), latest)
    except Exception:
        return False, None
