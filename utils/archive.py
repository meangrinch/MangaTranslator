import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from utils.logging import log_message

# CBZ comic archives are plain ZIP files with a different extension.
ARCHIVE_SUFFIXES = (".zip", ".cbz")

# BCP 47 tags for the UI's target languages, keyed by lowercased display name.
LANGUAGE_ISO_CODES = {
    "afrikaans": "af",
    "albanian": "sq",
    "arabic": "ar",
    "armenian": "hy",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese (simplified)": "zh-Hans",
    "chinese (traditional)": "zh-Hant",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "persian (farsi)": "fa",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "malay": "ms",
    "marathi": "mr",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian (cyrillic)": "sr-Cyrl",
    "serbian (latin)": "sr-Latn",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "filipino (tagalog)": "fil",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
}


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


def copy_comic_info(
    input_dir: Path, output_dir: Path, output_language: str, preserve_structure: bool
) -> None:
    """Copy ComicInfo.xml files to the output with LanguageISO set to the target."""
    candidates = input_dir.rglob("*") if preserve_structure else input_dir.iterdir()
    code = LANGUAGE_ISO_CODES.get(output_language.strip().lower())

    for src in candidates:
        if not (src.is_file() and src.name.lower() == "comicinfo.xml"):
            continue
        try:
            tree = ET.parse(src)
        except ET.ParseError as e:
            log_message(f"Skipping unreadable {src.name}: {e}", always_print=True)
            continue

        root = tree.getroot()
        language = root.find("LanguageISO")
        if code is None:
            # Keeping the source language would mislabel the translated pages.
            if language is not None:
                root.remove(language)
        else:
            if language is None:
                language = ET.SubElement(root, "LanguageISO")
            language.text = code

        dest = output_dir / src.relative_to(input_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tree.write(dest, encoding="utf-8", xml_declaration=True)
