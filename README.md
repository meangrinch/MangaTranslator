## MangaTranslator

Translate manga/comic speech bubbles using AI: YOLOv8 for bubble detection, LLMs for OCR+translation. Features a Gradio Web UI and a CLI.

## Features
- Speech bubble detection (YOLOv8)
- Text cleaning (removes original bubble text)
- LLM-powered translation (one-step or two-step)
- Text rendering (with custom font pack)
- Two interfaces: Web UI (Gradio) and CLI

## Requirements
- Python 3.10+
- PyTorch (CPU or CUDA build for your system)
- YOLO model trained for speech bubble detection (`.pt`)
- Font pack with `.ttf`/`.otf`
- Vision-capable LLM (API or Local)

## Installation

### Windows portable
Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/MangaTranslator/releases)
- Includes the recommended YOLO model and Komika/Comicka font packs

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
# Example (CUDA 12.6)
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
# Example (CPU)
pip install torch
```
4) Install dependencies
```bash
pip install -r requirements.txt
```

## Post-Install Setup
### YOLO model
- Place the necessary YOLO segmentation model (`.pt`) in `models/`
- Recommended Model: [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt)
- The application will automatically detect and use the model (no manual selection needed)

### Fonts
- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` so variants are detected
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
  --font-dir fonts/Komika \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:11434/v1 \
  --output ./output

# Cleaning-only mode (no translation/rendering)
python main.py --input <image_path> --cleaning-only

# Test mode (bypass translation; render placeholder lorem ipsum)
python main.py --input <image_path> --test-mode

# Full options
python main.py --help
```

## Web UI (Quick Start)
1) Launch the Web UI (use `start-webui.bat` on Windows, or the command above)
2) Translator tab (single) or Batch tab (multiple)
3) Upload image(s) and choose a font pack; set source/target languages
4) Open **Config** and set: LLM provider/model, API key or endpoint, reading direction (`rtl` for manga, `ltr` for comics)
5) Click **Translate** / **Start Batch Translating** — outputs save to `./output/`
6) Enable "Cleaning-only Mode" or "Test Mode" in **Other** to skip translation and/or render placeholder text

## Recommended Fonts

#### Normal Text
- CC Wild Words
- Anime Ace 3 BB
- Clementine
- Komika Hand

#### Shouts/SFX Text (Can be used in place of "Normal Text" bold/italic variants)
- Fold & Staple BB
- PP Handwriting
- Dirty Finger

## Troubleshooting

- **Wrong reading order:** Set correct "Reading Direction" (rtl for manga, ltr for comics)
- **Uncleaned text remaining:** Lower "Fixed Threshold Value" (e.g., 180) and/or reduce "Shrink Threshold ROI" (e.g., 0–2)
- **Outlines get eaten during cleaning:** Increase "Shrink Threshold ROI" (e.g., 6–8)
- **Incoherent translations/API refusals:** Try "two-step" translation mode for less-capable LLMs **or** Disable "Send Full Page to LLM" in Config → Translation
- **Inconsistent translations:** Adjust LLM parameters (e.g., "Temperature") for more creative/deterministic output
- **Text too large/small:** Adjust "Max Font Size" and "Min Font Size" ranges

## Updating
- Windows portable: download the latest release and replace the existing folder
- Manual install: from the repo root:
```bash
git pull
```

## License & credits
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)