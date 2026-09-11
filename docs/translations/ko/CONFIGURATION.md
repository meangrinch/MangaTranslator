# 설정

## 모델

필요한 모든 모델은 처음 사용할 때 자동으로 다운로드됩니다.

---

## LLM 설정

- **제공자:** Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible
- **Web UI:** Config 탭에서 제공자, 모델 및 API 키 설정 (로컬에 저장됨)
- **CLI:** 플래그 또는 환경 변수로 키/URL 전달
- **OpenAI-Compatible:** 로컬 엔드포인트(예: `http://localhost:8080/v1`) 및 Azure OpenAI 엔드포인트(예: `https://<resource>.openai.azure.com`) 지원

### 환경 변수

| 제공자 | 변수 |
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

### 텍스트 전용 모델 및 용어집

다음 모델은 OpenAI-Compatible 제공자를 통해 사용할 때 자동으로 감지되어 최적화된 프롬프트를 받습니다. 텍스트 전용 모델이며 2단계 번역 및 로컬 OCR이 필요합니다. `special_instructions` 필드는 해당 모델의 용어집 형식(한 줄에 하나의 항목, 예: `term -> translation`)에 매핑됩니다:

- **YanoljaNEXT-Rosetta** (예: `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`)
- **Hy-MT2** (예: `tencent/Hy-MT2-7B`), 권장 샘플링 매개변수도 자동으로 미리 채워집니다.

---

## 말풍선 외부 텍스트 설정 (선택 사항)

OSB(말풍선 외부) 텍스트 파이프라인을 사용하려면 게이트된 저장소에 대한 읽기 접근 권한이 있는 Hugging Face 토큰을 생성하세요:

1. [Hugging Face](https://huggingface.co/)에 로그인
2. 다음 저장소에서 이용 약관 동의:
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (선택 사항, Nunchaku 백엔드를 통해 FLUX.1 Kontext를 사용하는 경우)
   - [SAM 3](https://huggingface.co/facebook/sam3) (선택 사항, SAM 3을 사용하는 경우)
3. Hugging Face Settings에서 "Read access to contents of public gated repos" 권한이 있는 액세스 토큰 생성
4. MangaTranslator에 토큰 추가:
   - Web UI: Config에서 `hf_token` 설정 후 Save Config 클릭
   - CLI: `--osb-hf-token <...>` 전달
   - 환경 변수: `HF_TOKEN` 설정

---

## 폰트

`fonts/` 아래에 서브폴더를 만들고 `.otf` 또는 `.ttf` 파일을 넣습니다. 스타일 변형이 감지되도록 파일명에 `italic`, `bold`, 또는 `bolditalic`이 포함된 것을 권장합니다.

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

폰트 추천은 [폰트](FONTS.md)를 참조하세요.
