<p align="center">
  <a href="../../../README.md">English</a> |
  <a href="README.md">简体中文</a> |
  <a href="../ko/README.md">한국어</a> |
  <a href="../ja/README.md">日本語</a>
</p>

<h1 align="center"><b>MangaTranslator</b></h1>

<p align="center">
  <img src="https://img.shields.io/github/v/release/meangrinch/MangaTranslator?label=Release&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/downloads/meangrinch/MangaTranslator/total?label=Downloads&labelColor=181717&color=0877d2" />
  <img src="https://img.shields.io/github/license/meangrinch/MangaTranslator?labelColor=181717&color=2ea44f" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&labelColor=181717" />
</p>

<div align="center">
用于使用 AI 自动翻译漫画、条漫 (Manhwa) 和美漫的端到端工具。自动检测对话框与框外文本，利用扩散模型进行图像修复以擦除原文，并通过大语言模型 (LLM) 和自定义字体包实现 60 多种语言的自动排版翻译。
</div>

<br/>

<div align="center">
  <table>
    <tr>
      <th style="text-align: center">原文</th>
      <th style="text-align: center">翻译后（一键完成）</th>
    </tr>
    <tr>
      <td><img src="../../images/example_original.jpg" width="400" /></td>
      <td><img src="../../images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

---

## 快速上手

### 1. 下载

- **便携版（推荐）：** 从 [Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable) 下载便携版构建。
  - Windows： 无需额外系统要求。
  - Linux / macOS： 需要系统安装 Python 3.10+ 和 Git。
- **源码安装：**
  ```bash
  git clone https://github.com/meangrinch/MangaTranslator.git
  cd MangaTranslator
  python -m venv venv
  .\venv\Scripts\activate # 或 source venv/bin/activate (Linux/macOS)
  pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130 # 或 pip install torch==2.11.0 torchvision==0.26.0 (macOS)
  pip install -r requirements.txt
  ```

*更多信息请参阅 [安装](INSTALLATION.md)。*

### 2. 配置

- **服务商设置：** 在 Web UI 中，前往 Config → Translation 选择大语言模型服务商（例如 Google、OpenAI、Anthropic、DeepSeek 等）并输入 API 密钥，或使用本地 OpenAI 兼容端点（例如 llama.cpp）。保存配置以跨会话保留设置。
- **对话框外文本（可选）：** 前往 Config → OSB Text 并设置具有 [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) 访问权限的 Hugging Face 令牌。
- **环境变量：** 或者，可以在系统环境中配置凭据（例如 `GEMINI_API_KEY`、`HF_TOKEN`），同时适用于 Web UI 和命令行。

*更多信息请参阅 [配置](CONFIGURATION.md)。*

### 3. 翻译

- **Web UI：** 将图像上传到 Translator/Batch 选项卡，然后点击 Translate。
- **CLI：**
  ```bash
  python main.py --input "path/to/page.jpg" --input-language "Japanese" --output-language "English" --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>
  ```
*更多示例请参阅 [CLI](CLI.md)。*

---

## 功能特点

- **检测**：对话框与对话框外文本检测（YOLO、SAM 2.1/3）
- **擦除**：对话框与背景文本的重绘擦除（FLUX.2 Klein、FLUX.1 Kontext 或 OpenCV）
- **翻译**：支持 60 多种语言的 OCR 与翻译（云端 API 或本地 LLM）
- **渲染**：支持对齐、自动换行和自定义字体包的文本渲染引擎
- **超分辨率**：文本区域与整页原画超分辨率放大（2x-AnimeSharpV4）
- **处理**：支持目录结构保留的单图、文件夹和 ZIP 批量处理
- **配置**：灵活的配置选项，可适应多样的页面布局并精细调整输出质量
- **界面**：Web UI (Gradio) 和命令行界面 (CLI)
- **自动化**：一键翻译，无需人工干预

---

## 文档

- [硬件要求](HARDWARE_REQUIREMENTS.md)
- [安装](INSTALLATION.md)
- [配置](CONFIGURATION.md)
- [CLI](CLI.md)
- [字体](FONTS.md)
- [故障排除](TROUBLESHOOTING.md)

---

## 支持项目

MangaTranslator 是免费且开源的。如果它为您节省了时间或改善了阅读体验，欢迎考虑支持它的开发！

<p align="center">
  <a href="https://ko-fi.com/grinnch" target="_blank">
    <img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Support on Ko-fi" height="38"/>
  </a>
</p>

---

## 许可证和鸣谢

- 许可证：Apache-2.0（参见 [LICENSE](../../../LICENSE)）
- 作者：[grinnch](https://github.com/meangrinch)

<details>
<summary><b>ML 模型与相关开源库</b></summary>

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
