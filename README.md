<p align="center">
  <a href="README.md">English</a> |
  <a href="docs/translations/zh/README.md">简体中文</a> |
  <a href="docs/translations/ko/README.md">한국어</a> |
  <a href="docs/translations/ja/README.md">日本語</a>
</p>

<h1 align="center"><b>MangaTranslator</b></h1>

<p align="center">
  <img src="https://img.shields.io/github/v/release/meangrinch/MangaTranslator?label=Release&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/downloads/meangrinch/MangaTranslator/total?label=Downloads&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/license/meangrinch/MangaTranslator?labelColor=181717&color=2ea44f" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&labelColor=181717" />
</p>

<div align="center">
An end-to-end pipeline for translating manga, manhwas, and comics using AI. Detects speech bubbles and outside-bubble text, removes original text with diffusion inpainting, and typesets translations across 60+ languages using LLMs and custom fonts.
</div>

<br/>

<div align="center">
  <table>
    <tr>
      <th style="text-align: center">Original</th>
      <th style="text-align: center">Translated (One-Click)</th>
    </tr>
    <tr>
      <td><img src="docs/images/example_original.jpg" width="400" /></td>
      <td><img src="docs/images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

---

## Quick Setup

### 1. Download

- **Portable Package (Recommended):** Download the portable build from [Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable).
  - *Windows:* No requirements.
  - *Linux / macOS:* Requires Python 3.10+ and Git.
- **From Source:**
  ```bash
  git clone https://github.com/meangrinch/MangaTranslator.git
  cd MangaTranslator
  python -m venv venv
  # Activate: .\venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/macOS)
  pip install -r requirements.txt
  ```

### 2. Configure

- **Provider:** In the Web UI, navigate to Config → Translation to select your LLM provider (e.g., Google, OpenAI, Anthropic, DeepSeek...) and enter your API key, or use a local OpenAI-compatible endpoint (e.g., llama.cpp). Save to config to persist API keys/settings across sessions.
- **Outside-Bubble Text (Optional):** Navigate to Config → OSB Text and set a Hugging Face token with access to [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo).
- **Environment Variables:** Alternatively, you can configure credentials in your environment (e.g., `GEMINI_API_KEY`, `HF_TOKEN`) for both Web UI and CLI usage.

*For additional information, see [Configuration](docs/CONFIGURATION.md).*

### 3. Translate

- **Web UI:** Upload images to the Translator/Batch tab and click Translate.
- **CLI:**
  ```bash
  python main.py --input "path/to/page.jpg" --input-language "Japanese" --output-language "English" --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>
  ```
*For additional examples, see [CLI](docs/CLI.md).*

---

## Features

- **Detection**: Speech bubble and outside-bubble text detection (YOLO, SAM 2.1/3)
- **Cleaning**: Inpaint speech bubbles and background text (FLUX.2 Klein, FLUX.1 Kontext, or OpenCV)
- **Translation**: OCR and translation supporting 60+ languages (cloud API or local LLM)
- **Rendering**: Custom text rendering with alignment, word wrap, and custom font packs
- **Upscaling**: Text region and full-page artwork upscaling (2x-AnimeSharpV4)
- **Processing**: Single image, folder, and ZIP archive batch processing (with directory preservation)
- **Configuration**: Flexible controls to adapt to diverse page layouts and fine-tune output quality
- **Interfaces**: Web UI (Gradio) and CLI
- **Automation**: One-click translation; no manual intervention required

---

## Documentation

- [Hardware Requirements](docs/HARDWARE_REQUIREMENTS.md)
- [Installation](docs/INSTALLATION.md)
- [Configuration](docs/CONFIGURATION.md)
- [CLI](docs/CLI.md)
- [Fonts](docs/FONTS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## Support the Project

MangaTranslator is open-source and free. If it saves you time or enhances your reading experience, consider supporting its development!

<p align="center">
  <a href="https://ko-fi.com/grinnch" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Support on Ko-fi" height="38"/>
  </a>
</p>

---

## License & Credits

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)

<details>
<summary><b>ML Models and Libraries</b></summary>

- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 Speech Bubble Detector: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic Text and Bubble Detector RT-DETR-v2: [ogkalu](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- SAM 3: [Meta AI](https://huggingface.co/facebook/sam3)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- PaddleOCR-VL-1.6: [PaddlePaddle](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 4B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- FLUX.2 Klein 9B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- Nunchaku: [Nunchaku AI](https://github.com/nunchaku-ai/nunchaku)
- SDNQ Quants: [Disty0](https://huggingface.co/Disty0)
- Unsloth Quants: [Unsloth](https://huggingface.co/unsloth)
- stable-diffusion.cpp: [leejet](https://github.com/leejet/stable-diffusion.cpp)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)

</details>
