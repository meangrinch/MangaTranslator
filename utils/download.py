import os
import time
import urllib.request
from pathlib import Path

from utils.exceptions import ModelError
from utils.logging import log_message


def download_file(path: Path, url: str) -> None:
    """Download `url` to `path` unless it already exists.

    Writes to `<name>.part` first, so an interrupted download (including Ctrl+C)
    never leaves a truncated file at `path` that later runs would trust.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(path.name + ".part")
    log_message(f"Downloading {path.name}...", always_print=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(part_path, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_log = time.monotonic()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_log >= 5:
                    if total:
                        percent = min(100.0, downloaded * 100.0 / total)
                        log_message(
                            f"  - {path.name}: {downloaded / (1024**2):.1f}/"
                            f"{total / (1024**2):.1f} MiB ({percent:.1f}%)",
                            always_print=True,
                        )
                    else:
                        log_message(
                            f"  - {path.name}: {downloaded / (1024**2):.1f} MiB",
                            always_print=True,
                        )
                    last_log = now
        os.replace(part_path, path)
        log_message(f"Downloaded {path.name} successfully.", always_print=True)
    except Exception as e:
        raise ModelError(f"Failed to download {path.name}: {e}") from e
    finally:
        part_path.unlink(missing_ok=True)
