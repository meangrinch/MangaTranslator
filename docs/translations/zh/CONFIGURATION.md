# 配置

## 模型

所有必需的模型都会在首次使用时自动下载。

---

## 大语言模型设置

- **服务商：** Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible
- **Web UI：** 在 Config 选项卡中配置服务商、模型和 API 密钥（保存在本地）
- **CLI：** 通过参数或环境变量传递密钥/URL
- **OpenAI-Compatible：** 支持本地端点（例如 `http://localhost:8080/v1`）和 Azure OpenAI 端点（例如 `https://<resource>.openai.azure.com`）

### 环境变量

| 服务商 | 环境变量 |
| :--- | :--- |
| Google | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| SpaceXAI | `SPACEXAI_API_KEY` / `XAI_API_KEY` |
| Meta Model | `META_MODEL_API_KEY` / `META_API_KEY` |
| Z.ai | `ZAI_API_KEY` |
| Moonshot AI | `MOONSHOT_API_KEY` |
| Xiaomi MiMo | `MIMO_API_KEY` |
| QwenCloud | `QWENCLOUD_API_KEY` / `QWEN_API_KEY` |
| OpenCode | `OPENCODE_API_KEY` / `OPENCODE_ZEN_API_KEY` / `OPENCODE_GO_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI-Compatible | `OPENAI_COMPATIBLE_API_KEY` |

### 纯文本模型与术语表

通过 OpenAI-Compatible 服务商使用以下模型时会自动检测并获得优化的提示词。它们是纯文本模型，需要两步翻译模式加本地 OCR。`special_instructions` 字段映射到其术语表格式（每行一条，例如 `term -> translation`）：

- **YanoljaNEXT-Rosetta**（例如 `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`）
- **Hy-MT2**（例如 `tencent/Hy-MT2-7B`），同时会自动预填推荐的采样参数

---

## 对话框外文本设置（可选）

如需使用对话框外（OSB）文本处理管线，请创建具有受门控仓库读取权限的 Hugging Face 令牌：

1. 登录 [Hugging Face](https://huggingface.co/)
2. 在以下页面接受许可协议：
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)（可选，如果通过 Nunchaku 后端使用 FLUX.1 Kontext）
   - [SAM 3](https://huggingface.co/facebook/sam3)（可选，如果使用 SAM 3）
3. 在 Hugging Face 设置中，创建一个具有 "Read access to contents of public gated repos" 权限的访问令牌
4. 将令牌添加到 MangaTranslator：
   - Web UI：在 Config 中设置 `hf_token` 并点击 Save Config
   - CLI：传入 `--osb-hf-token <...>`
   - 环境变量：设置 `HF_TOKEN`

---

## 字体

将字体包作为子文件夹存放在 `fonts/` 目录下，包含 `.otf` 或 `.ttf` 文件。推荐在文件名中包含 `italic`、`bold` 或 `bolditalic` 以便自动识别变体。

```text
fonts/
├─ CC Wild Words/
│  ├─ CCWildWords-Regular.otf
│  ├─ CCWildWords-Italic.otf
│  ├─ CCWildWords-Bold.otf
│  └─ CCWildWords-BoldItalic.otf
└─ Komika Hand/
   ├─ KOMIKA-HAND.ttf
   └─ KOMIKA-HANDBOLD.ttf
```

有关推荐字体，请参阅 [字体](FONTS.md)。
