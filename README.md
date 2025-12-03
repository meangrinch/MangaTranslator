## MangaTranslator

Web application for automating the translation of manga/comic page images using AI. Targets speech bubbles and text outside of speech bubbles. Supports 54 languages and custom font pack usage.

## Features

- Speech bubble detection, segmentation, cleaning (YOLOv8 + SAM 2.1)
- Outside speech bubble text detection & inpainting (EasyOCR + Flux Kontext)
- LLM-powered OCR and translations (supports 54 languages)
- Text rendering (with custom font packs)
- Upscaling (2x-AnimeSharpV4)
- Single/Batch image processing with directory structure preservation and ZIP file support
- Two interfaces: Web UI (Gradio) and CLI

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, ROCm)
- YOLO model (`.pt`) for speech bubble detection; auto-downloaded
- Font pack with `.ttf`/`.otf`
- Any LLM for Japanese source text; vision-capable LLM for other languages (API or local)

## Install

### Windows portable

Download the standalone zip from the releases page: [Releases](https://github.com/meangrinch/MangaTranslator/releases)

- Default package: Download once, run `setup.bat` before first launch to install dependencies, and `update-standalone.bat` to update to the latest version. Installs `PyTorch v2.9.1+cu128`.
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

4. Install Nunchaku (optional, for inpainting outside-bubble text.)

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

- If you want to use the OSB text pipeline, you need a Hugging Face token with access to FLUX.1 Kontext.
- Follow these steps to create one:

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
# Single image, Japanese → English, Google provider
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...>

# Batch folder, custom source/target languages, OpenAI-Compatible provider (LM Studio)
python main.py --input <folder_path> --batch \
  --font-dir "fonts/Komika" \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:1234/v1 \
  --output ./output

# Single Image, Japanese → English (Google), OSB text detection, custom OSB font
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

- **Incorrect reading order:**
  - Set correct "Reading Direction" (rtl for manga, ltr for comics)
  - Try "two-step" translation mode for less-capable LLMs
- **Uncleaned text remaining:**
  - Lower "Fixed Threshold Value" (e.g., 180) and/or reduce "Shrink Threshold ROI" (e.g., 0–2)
- **Outlines get eaten during cleaning:**
  - Increase "Shrink Threshold ROI" (e.g., 6–8)
- **Conjoined bubbles not detected:**
  - Ensure "Detect Conjoined Bubbles" is enabled
  - Lower "Bubble Detection Confidence" (e.g., 0.20)
- **Incoherent translations/API refusals:**
  - Try "two-step" translation mode for less-capable LLMs
  - Try disabling "Send Full Page to LLM"
  - Try using "manga-ocr" OCR method, particularly for less-capable LLMs (Japanese sources only)
  - Increase "max_tokens" and/or use a higher "reasoning_effort" (e.g., "high")
- **Text too large/small:**
  - Adjust "Max Font Size" and "Min Font Size" ranges
- **Inconsistent behavior across different image sizes:**
  - Enable "Auto-Scale to Image Size": Settings must be tuned for 1MP sized images for proper scaling (default values should suffice)
- **OSB text not inpainted/cleaned:**
  - Ensure "Enable OSB Detection" is enabled
  - Install Nunchaku (see Installation step 4)
  - Set your Hugging Face access token (see Post-Install Setup)
- **High LLM token usage:**
  - Disable "Send Full Page to LLM"
  - Lower "Bubble Min Side Pixels"/"Context Image Max Side Pixels"/"OSB Min Side Pixels" target sizes
  - Lower "Media Resolution" (if using Gemini models)

## Updating

- Windows portable:
  - Default Package: Run `update-standalone.bat`
  - Pre-downloaded Package: Download the latest version from the releases page
- Manual install: from the repo root:

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
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
