## MangaTranslator

Translate manga/comic speech bubbles using AI: YOLOv8 for bubble detection, LLMs for OCR+translation. Features a Gradio Web UI and a CLI.

## Features
- Speech bubble detection (YOLOv8)
- Text cleaning (removes original bubble text)
- LLM-powered OCR and translation
- Text rendering (with custom font packs)
- Two interfaces: Web UI (Gradio) and CLI

## Requirements
- Python 3.10+
- PyTorch (CPU or CUDA build for your system)
- YOLO model (`.pt`) for speech bubble detection; auto-downloaded
- Font pack with `.ttf`/`.otf`
- Vision-capable LLM (API or local)

## Installation

### Windows portable
Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/MangaTranslator/releases)
- Includes the Komika (for normal text) and Cookies (for OSB text) font packs
- Run the `setup.bat` file before first launch to install dependencies

### Manual install
1) Clone and enter the repo
```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```
2) Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```
3) Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))
```bash
# Example (CUDA 12.8)
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
# Example (CPU)
pip install torch
```
4) Install Nunchaku (optional, for inpainting outside-bubble text.)
- Nunchaku wheels are not on PyPI. Install directly from the v1.0.1 GitHub release URL, matching your OS and Python version.
```bash
# Example (Windows, Python 3.13, PyTorch 2.7.1)
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.7-cp313-cp313-win_amd64.whl
```
5) Install dependencies
```bash
pip install -r requirements.txt
```

## Post-Install Setup
### Models
- The application will automatically download and use all required models

### Fonts
- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` or `italic-bold` so variants are detected
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
- Providers: Google, OpenAI, Anthropic, xAI, OpenRouter, OpenAI-Compatible
- Web UI: configure provider/model/key in the Config tab (stored locally)
- CLI: pass keys/URLs as flags or via env vars
- Env vars: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-compatible default URL: `http://localhost:11434/v1`

### OSB text setup (optional)
If you want to use the OSB text pipeline, you need a Hugging Face token with access to FLUX.1 Kontext.

Follow these steps to create one:

1. Sign in or create a Hugging Face account
2. Visit and accept the terms on: [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
3. Create a new access token in your Hugging Face settings with read access to gated repos ("Read access to contents of public gated repos")
4. Add the token to the app:

   - Web UI: set `hf_token` in Config
   - Env var (alternative): set `HUGGINGFACE_TOKEN`

## Run

### Web UI (Gradio)
- Windows: double-click `start-webui.bat` (`venv` must be present)
- Or run:
```bash
python app.py --open-browser
```
Options: `--models` (default `./models`), `--fonts` (default `./fonts`), `--port` (default `7676`), `--cpu`.
First launch can take ~1–2 minutes.

### CLI
Examples:
```bash
# Single image, Japanese → English (Google)
python main.py --input <image_path> \
  --provider Google --google-api-key <AI...>

# Batch folder, custom languages (OpenAI-Compatible, e.g., Ollama)
python main.py --input <folder_path> --batch \
  --font-dir "fonts/Komika" \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:11434/v1 \
  --output ./output

# Single Image, Japanese → English (Google), outside speech bubble text detection, custom OSB font
python main.py --input <image_path> \
  --provider Google --google-api-key <AI...> \
  --osb-enable --osb-font-name "fonts/fast_action"

# Cleaning-only mode (no translation/rendering)
python main.py --input <image_path> --cleaning-only

# Test mode (bypass translation; render placeholder lorem ipsum)
python main.py --input <image_path> --test-mode

# Full options
python main.py --help
```

## Web UI (Quick Start)
1) Launch the Web UI (use `start-webui.bat` on Windows, or the command above)
2) Upload image(s) in the Translator tab (single) or Batch tab (multiple)
3) Choose a font pack; set source/target languages
4) Open **Config** and set: LLM provider/model, API key or endpoint, reading direction (`rtl` for manga, `ltr` for comics)
5) Click **Translate** / **Start Batch Translating** — outputs save to `./output/`
6) Enable "Cleaning-only Mode" or "Test Mode" in **Other** to skip translation and/or render placeholder text

## Recommended Fonts

#### Normal Text
- CC Wild Words
- Anime Ace 3 BB
- CC Astro City Int
- Clementine
- Komika Hand

#### Shouts/SFX Text (Can be used in place of "Normal Text" bold/italic variants)
- Fold & Staple BB
- PP Handwriting
- Dirty Finger

#### Outside Speech Bubble Text
- Fast Action
- Shark Soft Bites
- Cookies
- Yankareshi

## Troubleshooting

- **Wrong reading order:** Set correct "Reading Direction" (rtl for manga, ltr for comics)
- **Uncleaned text remaining:** Lower "Fixed Threshold Value" (e.g., 180) and/or reduce "Shrink Threshold ROI" (e.g., 0–2)
- **Outlines get eaten during cleaning:** Increase "Shrink Threshold ROI" (e.g., 6–8)
- **Incoherent translations/API refusals:** Try "two-step" translation mode for less-capable LLMs **or** Disable "Send Full Page to LLM" in Config → Translation
- **Inconsistent translations:** Adjust LLM parameters (e.g., "Temperature") for more creative/deterministic output
- **Text too large/small:** Adjust "Max Font Size" and "Min Font Size" ranges
- **OSB text not inpainted/cleaned:** Ensure "Outside Speech Bubble" is enabled, install Nunchaku (see Installation step 4), and set your Hugging Face token (`hf_token`).

## Updating
- Windows portable: run `update-standalone.bat` or download the latest release and replace the existing folder
- Manual install: run `update.bat` or from the repo root:
```bash
git pull
```

## License & credits
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)

#### ML Models
- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Comic Speech Bubble Detector YOLOv8m: [ogkalu](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
- SAM 2.1 (Segment Anything): [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- EasyOCR: [JaidedAI](https://github.com/JaidedAI/EasyOCR)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- Nunchaku: [Nunchaku Tech](https://github.com/nunchaku-tech/nunchaku)
- 2x-AnimeSharpV4 RCAN: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)