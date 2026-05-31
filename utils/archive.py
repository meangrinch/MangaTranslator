from pathlib import Path
from zipfile import ZipFile


def safe_extract_zip(zip_ref: ZipFile, destination: Path) -> None:
    destination = Path(destination).resolve()

    for member in zip_ref.infolist():
        target = destination / member.filename
        try:
            target.resolve().relative_to(destination)
        except ValueError as exc:
            raise ValueError(
                f"ZIP archive contains an unsafe path: {member.filename}"
            ) from exc

    zip_ref.extractall(destination)
