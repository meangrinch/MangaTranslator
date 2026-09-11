# 設定

## モデル

必要なモデルはすべて初回使用時に自動的にダウンロードされます。

---

## LLM設定

- **プロバイダー：** Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible
- **Web UI：** Configタブでプロバイダー、モデル、APIキーを設定（ローカルに保存）
- **CLI：** フラグまたは環境変数経由でキー/URLを渡す
- **OpenAI-Compatible：** ローカルエンドポイント（例：`http://localhost:8080/v1`）および Azure OpenAI エンドポイント（例：`https://<resource>.openai.azure.com`）に対応

### 環境変数

| プロバイダー | 変数名 |
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

### テキスト専用モデルと用語集

以下のモデルは、OpenAI-Compatibleプロバイダー経由で使用すると自動検出され、最適化されたプロンプトが適用されます。テキスト専用モデルのため、2段階翻訳モードとローカルOCRが必要です。`special_instructions` フィールドは用語集フォーマット（1行に1項目、例：`term -> translation`）にマッピングされます：

- **YanoljaNEXT-Rosetta**（例：`yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`）
- **Hy-MT2**（例：`tencent/Hy-MT2-7B`、推奨サンプリングパラメータも自動入力されます）

---

## 吹き出し外テキスト設定（任意）

OSB（吹き出し外）テキスト処理パイプラインを使用するには、ゲート付きリポジトリへの読み取りアクセス権を持つHugging Faceトークンを作成してください：

1. [Hugging Face](https://huggingface.co/) にログイン
2. 以下のリポジトリで利用規約に同意：
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)（任意、Nunchakuバックエンド経由でFLUX.1 Kontextを使用する場合）
   - [SAM 3](https://huggingface.co/facebook/sam3)（任意、SAM 3を使用する場合）
3. Hugging FaceのSettingsで、「Read access to contents of public gated repos」権限を持つアクセストークンを作成
4. MangaTranslatorにトークンを追加：
   - Web UI：Configで `hf_token` を設定し、「Save Config」をクリック
   - CLI：`--osb-hf-token <...>` を指定
   - 環境変数：`HF_TOKEN` を設定

---

## フォント

`fonts/` 配下にサブフォルダを作成し、その中に `.otf` または `.ttf` ファイルを配置します。フォントのバリエーションが認識されるよう、ファイル名に `italic`、`bold`、または `bolditalic` を含めることを推奨します。

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

フォントの提案については [フォント](FONTS.md) を参照してください。
