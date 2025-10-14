from __future__ import annotations

import json
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import Version

API_URL = "https://api.github.com/repos/{repo}/releases/latest"


def get_latest_release_tag(repo: str, timeout: float = 3.0) -> Optional[str]:
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
) -> Tuple[bool, Optional[str]]:
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
