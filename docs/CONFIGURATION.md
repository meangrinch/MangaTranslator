# Configuration

## Models

All required models are downloaded automatically on first use.

---

## LLM Setup

- **Providers:** Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible
- **Web UI:** Configure provider, model, and API key in the Config tab (stored locally)
- **CLI:** Pass keys/URLs as flags or via environment variables
- **OpenAI-Compatible:** Supports local endpoints (e.g., `http://localhost:8080/v1`) and Azure OpenAI endpoints (e.g., `https://<resource>.openai.azure.com`)

### Environment Variables

| Provider | Variable |
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

### Text-Only Models & Glossaries

The following models are automatically detected when used via the OpenAI-Compatible provider and receive optimized prompting. They are text-only and require two-step translation + local OCR. The `special_instructions` field maps to their glossary format (one entry per line, e.g., `term -> translation`):

- **YanoljaNEXT-Rosetta** (e.g., `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`)
- **Hy-MT2** (e.g., `tencent/Hy-MT2-7B`), which also pre-fills recommended sampling parameters

---

## OSB Text Setup (Optional)

To use the OSB (outside speech bubble) text pipeline, create a Hugging Face token with read access to gated repositories:

1. Sign in to [Hugging Face](https://huggingface.co/)
2. Accept the terms on:
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (optional, if using FLUX.1 Kontext via Nunchaku backend)
   - [SAM 3](https://huggingface.co/facebook/sam3) (optional, if using SAM 3)
3. In Hugging Face Settings, create an access token with "Read access to contents of public gated repos"
4. Add the token to MangaTranslator:
   - Web UI: Set `hf_token` in Config and click Save Config
   - CLI: Pass `--osb-hf-token <...>`
   - Env var: Set `HF_TOKEN`

---

## Fonts

Put font packs as subfolders in `fonts/` with `.otf` or `.ttf` files. Prefer filenames that include `italic`, `bold`, or `bolditalic` so variants are detected.

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

For font suggestions, see [Fonts](FONTS.md).
