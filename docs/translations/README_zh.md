[English](../../README.md) | [简体中文](README_zh.md) | [한국어](README_ko.md) | [日本語](README_ja.md)

## MangaTranslator

基于 Gradio 的 Web 应用程序，用于使用 AI 自动翻译漫画/动漫页面图像。针对对话框（气泡框）和对话框外的文本。支持 59 种语言，并支持自定义字体包。

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">原文</th>
      <th style="text-align: left">翻译后（一键完成）</th>
    </tr>
    <tr>
      <td><img src="../images/example_original.jpg" width="400" /></td>
      <td><img src="../images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

## 目录

- [功能特点](#功能特点)
- [运行要求](#运行要求)
- [安装](#安装)
- [安装后设置](#安装后设置)
- [运行](#运行)
- [文档](#文档)
- [更新](#更新)
- [许可证和鸣谢](#许可证和鸣谢)

## 功能特点

- **检测**：对话框（气泡框）检测与分割（YOLO + SAM 2.1/3）
- **擦除**：擦除对话框和对话框外 (OSB) 的文本（Flux.2 Klein、Flux.1 Kontext 或 OpenCV）
- **翻译**：基于大语言模型 (LLM) 的 OCR 与翻译（支持 59 种语言）
- **渲染**：支持排版对齐和自定义字体包的文本渲染
- **超分辨率**：使用 2x-AnimeSharpV4 提升输出图像质量
- **处理**：支持目录结构保留与 ZIP 压缩包的单张/批量处理
- **界面**：Web UI (Gradio) 和命令行界面 (CLI)
- **自动化**：一键翻译，无需人工干预

## 运行要求

- Python 3.10+
- PyTorch (支持 CPU、CUDA、ROCm、XPU、MPS)
- 包含 `.ttf`/`.otf` 文件的字体包；便携版已包含
- 日文原文需使用大语言模型（LLM）；其他语言需使用多模态模型（VLM）（支持 API 或本地部署）

## 安装

### 便携版（推荐）

从 Releases 页面下载独立压缩包：[Portable Build (便携版构建)](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)

**系统要求：**

- **Windows：** 已捆绑 Python/Git；无其他额外系统要求
- **Linux/macOS：** 系统中必须安装有 Python 3.10+ 和 Git

**设置方法：**

1. 解压 zip 压缩包
2. 运行对应平台的设置脚本：
   - **Windows：** 双击 `setup.bat`
   - **Linux/macOS：** 在终端中运行 `./setup.sh`
3. 系统将自动检测并安装适配您系统的 PyTorch 版本
4. 运行 `./MangaTranslator/` 目录下的启动脚本：
   - **Windows：** 双击 `start-webui.bat`
   - **Linux/macOS：** 运行 `start-webui.sh`

包含的字体包：

- _Komika_（普通文本）
- _Cookies_（对话框外文本）
- _Comicka_（两者皆可）
- _Roboto_（支持变音符号）
- _Noto Sans SC_（支持简体中文）

> [!TIP]
> 如果您需要迁移到新的便携版：
>
> - 您可以安全地将 `fonts`、`models` 和 `output` 目录移动到新的便携版中
> - 在设置配置相同的情况下，您也可以直接迁移 `runtime` 目录

### 手动安装

1. 克隆并进入仓库

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

2. 创建并激活虚拟环境（推荐）

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. 安装 PyTorch（请参阅：[PyTorch 安装指南](https://pytorch.org/get-started/locally/)）

```bash
# 示例 (CUDA 13.0)
pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
# 示例 (ROCm 7.1)
pip install torch==2.10.0+rocm7.1 torchvision==0.25.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1
# 示例 (XPU)
pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu
# 示例 (MPS/CPU)
pip install torch==2.10.0 torchvision==0.25.0
```

4. 安装 Nunchaku（可选，用于 Flux.1 Kontext Nunchaku 后端）

- Nunchaku 的 wheel 包未发布在 PyPI 上。请直接从 v1.2.1 GitHub release URL 安装适配您系统和 Python 版本的包。仅支持 CUDA，且需要 2000 系列及以上的显卡。

```bash
# 示例 (Windows, Python 3.13, PyTorch 2.10.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp313-cp313-win_amd64.whl
```

> [!NOTE]
> 通过 SDNQ 后端使用 Flux 模型时，不需要安装 Nunchaku。

5. 安装依赖项

```bash
pip install -r requirements.txt
```

## 安装后设置

### 模型 (Models)

- 应用程序将自动下载并使用所有必需的模型

### 字体 (Fonts)

- 将字体包作为子文件夹存放在 `fonts/` 目录下，其中应包含 `.otf`/`.ttf` 文件
- 字体文件名中最好包含 `italic`（斜体）/`bold`（粗体）或两者兼有，以便程序自动检测字形变体
- 示例目录结构：

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

### 大语言模型设置 (LLM Setup)

- 服务商支持：Google, OpenAI, Anthropic, xAI, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, OpenRouter, OpenAI-Compatible (OpenAI兼容)
- Web UI：在 Config (配置) 选项卡中配置服务商、模型和 API 密钥（将保存在本地）
- CLI：通过命令行参数或环境变量传递密钥/URL
- 环境变量：`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `MIMO_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI 兼容的默认 URL：`http://localhost:8080/v1`

> [!NOTE]
> 当通过 OpenAI 兼容服务商使用 YanoljaNEXT-Rosetta 模型（例如 `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`）时，程序会自动检测并使用优化的提示词。这些是纯文本模型，需要启用两步翻译模式和本地 OCR 模型。特殊说明 (Special Instructions) 字段将被映射为 Rosetta 的翻译词汇表（每行一个条目，例如 `Yanolja NEXT -> 雅놀자넥스트`）。

### 对话框外 (OSB) 文本设置（可选）

如果您希望使用对话框外（气泡外）文本擦除与重绘管线，您需要拥有一个对以下 Hugging Face 仓库具有访问权限的 Token：

- `deepghs/AnimeText_yolo`

#### 创建 Token 的步骤：

1. 登录或创建一个 Hugging Face 账号
2. 访问并接受以下仓库的许可协议：
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)（可选，如果使用带有 Nunchaku 的 Kontext）
   - [SAM 3](https://huggingface.co/facebook/sam3)（可选，如果使用 SAM 3）
3. 在 Hugging Face 设置中创建一个新的 Access Token，并赋予对受门控仓库的只读权限（"Read access to contents of public gated repos"）
4. 将 Token 添加到应用程序中：
   - Web UI：在 Config 中设置 `hf_token`
   - 环境变量（替代方案）：设置 `HUGGINGFACE_TOKEN`
5. 保存配置以在以后的会话中保留 Token

## 运行

### Web UI (Gradio)

- **便携版：**
  - Windows：双击 `MangaTranslator` 目录下的 `start-webui.bat`
  - Linux/macOS：在终端运行 `MangaTranslator` 目录下的 `./start-webui.sh`
- **手动安装：**
  - Windows/Linux/macOS：运行 `python app.py --open-browser`

可选参数：`--models`（默认 `./models`）、`--fonts`（默认 `./fonts`）、`--port`（默认 `7676`）、`--cpu`。
首次启动可能需要大约 1–2 分钟。

启动后，在 Config 选项卡中配置您的大语言模型服务商，然后上传图片并点击 Translate。

### CLI (命令行界面)

使用示例：

```bash
# 单张图片，日译英，使用 Google 服务商
python main.py --input <图片路径> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <API密钥>

# 批量文件夹，自定义源/目标语言，使用 OpenAI 兼容服务商 (llama.cpp)
python main.py --input <文件夹路径> --batch \
  --font-dir "fonts/Komika" \
  --input-language <源语言> --output-language <目标语言> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output

# 单张图片，日译英 (Google)，启用对话框外文本管线，使用自定义的对话框外文本字体
python main.py --input <图片路径> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <API密钥> \
  --osb-enable --osb-font-dir "fonts/Clementine"

# 仅擦除模式（不进行翻译/文本渲染）
python main.py --input <图片路径> --cleaning-only

# 仅超分辨率模式（不进行检测/翻译，仅进行放大）
python main.py --input <图片路径> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# 测试模式（不进行实际翻译；渲染占位符文本）
python main.py --input <图片路径> --test-mode

# 查看全部选项
python main.py --help
```

## 文档

- [硬件运行要求](../HARDWARE_REQUIREMENTS.md)
- [推荐字体](../FONTS.md)
- [常见问题与故障排除](../TROUBLESHOOTING.md)

## 更新

### 便携版

- Windows：运行便携版根目录下的 `update.bat`
- Linux/macOS：运行便携版根目录下的 `./update.sh`

### 手动安装

在仓库根目录下运行：

```bash
git pull
pip install -r requirements.txt  # 如果有虚拟环境，请先激活虚拟环境
```

## 许可证和鸣谢

- 许可证：Apache-2.0 (见 [LICENSE](../../LICENSE))
- 作者：[grinnch](https://github.com/meangrinch)

<details>
<summary><b>机器学习模型与库</b></summary>

- YOLOv8m 对话框检测器: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 对话框检测器: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic 对话框检测器 YOLOv8m: [ogkalu](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: Segment Anything in Images and Videos: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- SAM 3: [Meta AI](https://huggingface.co/facebook/sam3)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 4B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- FLUX.2 Klein 9B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- Nunchaku: [Nunchaku AI](https://github.com/nunchaku-ai/nunchaku)
- SDNQ Quants: [Disty0](https://huggingface.co/Disty0)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- PaddleOCR-VL-1.5: [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR-VL-1.5)

</details>
