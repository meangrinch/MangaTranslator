## MangaTranslator

Gradio-based web application for automating the translation of manga/comic page images using AI. Targets speech bubbles and text outside of speech bubbles. Supports 54 languages and custom font pack usage.

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">Original</th>
      <th style="text-align: left">Translated (w/ a single click)</th>
    </tr>
    <tr>
      <td><img src="docs/images/example_original.jpg" width="400" /></td>
      <td><img src="docs/images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

## Features

- Speech bubble detection, segmentation, cleaning (YOLO + SAM 2.1)
- Outside speech bubble text detection & inpainting (YOLO + Flux Kontext/OpenCV)
- LLM-powered OCR and translations (supports 54 languages)
- Text rendering and alignment (with custom font packs)
- Upscaling (2x-AnimeSharpV4)
- Single/Batch image processing with directory structure preservation and ZIP file support
- Two interfaces: Web UI (Gradio) and CLI
- All-in-one button; no human intervention required
- Various options to tailor the process

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, ROCm)
- YOLO model (`.pt`) for speech bubble detection; auto-downloaded
- Font pack with `.ttf`/`.otf`
- Any LLM for Japanese source text; vision-capable LLM for other languages (API or local)

## Install

### Windows portable

Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/MangaTranslator/releases)

- Default package: Download once, run `setup.bat` before first launch to install dependencies, and `update-standalone.bat` to update to the latest version (see [Updating](#updating)). Installs `PyTorch v2.9.1+cu128`.
- Pre-downloaded package: Download per version, no setup required, and no included update script. Contains `PyTorch v2.9.1+cu128`.
- Both include the Komika (for normal text), Cookies (for OSB text), and Comicka (for either) font packs

### Manual install

1. Clone and enter the repo

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))

```bash
# Example (CUDA 12.8)
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# Example (CPU)
pip install torch torchvision
```

4. Install Nunchaku (optional, for inpainting OSB text regions with Flux Kontext; not required for OpenCV inpainting)

- Nunchaku wheels are not on PyPI. Install directly from the v1.0.2 GitHub release URL, matching your OS and Python version. CUDA only.

```bash
# Example (Windows, Python 3.13, PyTorch 2.9.1)
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.9-cp313-cp313-win_amd64.whl
```

5. Install dependencies

```bash
pip install -r requirements.txt
```

## Post-Install Setup

### Models

- The application will automatically download and use all required models

### Fonts

- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` or both so variants are detected
- Example structure:

```text
fonts/
├─ CC Wild Words/
│  ├─ CCWildWords-Regular.otf
│  ├─ CCWildWords-Italic.otf
│  ├─ CCWildWords-Bold.otf
│  └─ CCWildWords-BoldItalic.otf
└─ Komika/
   ├─ KOMIKA-HAND.ttf
   └─ KOMIKA-HANDBOLD.ttf
```

### LLM setup

- Providers: Google, OpenAI, Anthropic, xAI, DeepSeek, Z.ai, Moonshot AI, OpenRouter, OpenAI-Compatible
- Web UI: configure provider/model/key in the Config tab (stored locally)
- CLI: pass keys/URLs as flags or via env vars
- Env vars: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-compatible default URL: `http://localhost:1234/v1`

### OSB text setup (optional)

If you want to use the OSB text pipeline, you need a Hugging Face token with access to the following repositories:

- `deepghs/AnimeText_yolo`
- `black-forest-labs/FLUX.1-Kontext-dev` (only required if using Flux Kontext)

#### Steps to create a token:

1. Sign in or create a Hugging Face account
2. Visit and accept the terms on: [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) (and [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) if applicable)
3. Create a new access token in your Hugging Face settings with read access to gated repos ("Read access to contents of public gated repos")
4. Add the token to the app:
   - Web UI: set `hf_token` in Config
   - Env var (alternative): set `HUGGINGFACE_TOKEN`
5. Save config to preserve the token across sessions

## Run

### Web UI (Gradio)

- Windows: double-click `start-webui.bat` (`venv` must be present for manual install)
- Or run:

```bash
python app.py --open-browser
```

Options: `--models` (default `./models`), `--fonts` (default `./fonts`), `--port` (default `7676`), `--cpu`.
First launch can take ~1–2 minutes.

### CLI

Examples:

```bash
# Single image, Japanese → English, Google provider
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...>

# Batch folder, custom source/target languages, OpenAI-Compatible provider (LM Studio)
python main.py --input <folder_path> --batch \
  --font-dir "fonts/Komika" \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:1234/v1 \
  --output ./output

# Single Image, Japanese → English (Google), OSB text pipeline, custom OSB text font
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...> \
  --osb-enable --osb-font-name "fonts/fast_action"

# Cleaning-only mode (no translation/text rendering)
python main.py --input <image_path> --cleaning-only

# Test mode (no translation; render placeholder text)
python main.py --input <image_path> --test-mode

# Full options
python main.py --help
```

## Web UI (Quick Start)

1. Launch the Web UI (use `start-webui.bat` on Windows, or the command above)
2. Upload image(s) in the Translator tab (single) or Batch tab (multiple)
3. Choose a font pack; set source/target languages
4. Open **Config** and set: LLM provider/model, API key or endpoint, reading direction (`rtl` for manga, `ltr` for comics)
5. Click **Translate** / **Start Batch Translating** — outputs save to `./output/`
6. Enable "Cleaning-only Mode" or "Test Mode" in **Other** to skip translation and/or render placeholder text

## Documentation

- [Recommended Fonts](docs/FONTS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Updating

- Windows portable:
  - Default Package: Run `update-standalone.bat`. To update requirements, run `update-standalone.bat` -> `setup.bat`
  - Pre-downloaded Package: Download the latest version from the releases page
- Manual install: from the repo root:

```bash
git pull
# Update requirements if needed
venv\Scripts\activate  # If venv present
pip install -r requirements.txt
```

## License & credits

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)
- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Comic Speech Bubble Detector YOLOv8m: [ogkalu](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
- SAM 2.1 (Segment Anything): [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- Nunchaku: [Nunchaku Tech](https://github.com/nunchaku-tech/nunchaku)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
