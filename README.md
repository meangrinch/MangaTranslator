## MangaTranslator

Translate manga/comic speech bubbles using AI: YOLOv8‑seg for bubble detection, LLMs for OCR+translation. Features a Gradio Web UI and a CLI.

## Features
- Speech bubble detection (YOLOv8‑seg)
- Text cleaning (removes original bubble text)
- LLM-powered translation (one-step or two-step)
- Text rendering with your chosen font pack
- Two interfaces: Web UI (Gradio) and CLI

## Requirements
- Python 3.10+
- PyTorch (CPU or CUDA build for your system)
- Segmentation YOLO model trained for speech bubbles (`.pt`)
- Font pack with `.ttf`/`.otf`
- Vision-capable LLM: Gemini, OpenAI, Anthropic, OpenRouter, or OpenAI-compatible (e.g., Ollama/LM Studio)

## Install

### Windows portable
Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/MangaTranslator/releases)
- Includes the recommended YOLO model and Komika font pack

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
# Example (CUDA 12.4)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
# Example (CPU)
pip install torch
```
4) Install dependencies
```bash
pip install -r requirements.txt
```

## Post-install setup
### YOLO model
- Place a segmentation model in `models/` (e.g., `models/yolov8m_seg-speech-bubble.pt`)
- Recommended: [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt)

### Fonts
- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` so variants are detected
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
- Providers: Gemini, OpenAI, Anthropic, OpenRouter, OpenAI-Compatible
- Web UI: configure provider/model/key in the Config tab (stored locally)
- CLI: pass keys/URLs as flags or via env vars
- Optional env vars: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-compatible default URL: `http://localhost:11434/v1`

## Run

### Web UI (Gradio)
- Windows: double-click `start-webui.bat`
- Or run:
```bash
python app.py --open-browser
```
Options: `--models` (default `./models`), `--fonts` (default `./fonts`), `--port` (default `7676`), `--cpu`.
First launch can take ~1–2 minutes.

### CLI
Examples:
```bash
# Single image, Japanese → English (Gemini)
python main.py --input <image_path> \
  --yolo-model models/yolov8m_seg-speech-bubble.pt \
  --provider Gemini --gemini-api-key <AI...>

# Batch folder, custom languages (OpenAI-Compatible, e.g., Ollama)
python main.py --input <folder_path> --batch \
  --yolo-model models/yolov8m_seg-speech-bubble.pt \
  --font-dir fonts/Komika \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:11434/v1 \
  --output ./output

# Cleaning only (no translation/rendering)
python main.py --input <image_path> \
  --yolo-model models/yolov8m_seg-speech-bubble.pt \
  --cleaning-only

# Full options
python main.py --help
```

## Web UI (quick steps)
1) Launch the Web UI (use `start-webui.bat` on Windows, or the command above)
2) Translator tab (single) or Batch tab (multiple)
3) Upload image(s) and choose a font pack; set source/target languages
4) Open **Config** and set: LLM provider/model, API key or endpoint, reading direction (`rtl` for manga, `ltr` for comics)
5) Click **Translate** / **Start Batch Translating** — outputs save to `./output/`
6) Use "Cleaning Only" in **Other** to skip translation/rendering

## Troubleshooting
- Models/fonts not found: use Config → Refresh; ensure `models/*.pt` and `fonts/<pack>/*.ttf|*.otf`
- OpenAI-compatible models not listed: verify the base URL (e.g., `http://localhost:11434/v1`) and that the endpoint is running
- GPU/CPU: CUDA used if available; add `--cpu` to force CPU
- Minimum image size: 600×600
- No bubbles detected: lower `--conf` and confirm the model supports segmentation

## Updating
```bash
git pull
```

## License & credits
- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)