[English](../../../README.md) | [简体中文](../zh/README.md) | [한국어](../ko/README.md) | [日本語](README.md)

## MangaTranslator

AIを使用して漫画やコミックページの画像翻訳を自動化する、GradioベースのWebアプリケーションです。吹き出し内のテキストと吹き出し外のテキストの両方を翻訳対象とします。60言語およびカスタムフォントパックの使用に対応しています。

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">オリジナル</th>
      <th style="text-align: left">翻訳後（ワンクリックで完了）</th>
    </tr>
    <tr>
      <td><img src="../../images/example_original.jpg" width="400" /></td>
      <td><img src="../../images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

## 目次

- [主な機能](#主な機能)
- [システム要件](#システム要件)
- [インストール](#インストール)
- [インストール後のセットアップ](#インストール後のセットアップ)
- [実行](#実行)
- [ドキュメント](#ドキュメント)
- [アップデート](#アップデート)
- [ライセンスとクレジット](#ライセンスとクレジット)

## 主な機能

- **検出**: 吹き出しの検出とセグメンテーション（YOLO、SAM 2.1/3）
- **クリーニング**: 吹き出し内および吹き出し外（OSB）テキストのインペインティング（FLUX.2 Klein、FLUX.1 Kontext、またはOpenCV）
- **翻訳**: LLMを活用したOCRと翻訳（60言語対応）
- **レンダリング**: 位置合わせやカスタムフォントパックをサポートしたカスタムテキスト描画エンジン
- **アップスケーリング**: テキスト領域およびページ全体のイラストのアップスケーリング（2x-AnimeSharpV4）
- **処理**: ディレクトリ構造を維持した、単一画像またはバッチ処理（ZIP対応）
- **設定**: 多様なページレイアウトに対応し、出力品質を微調整するための柔軟な設定項目
- **インターフェース**: Web UI（Gradio）およびCLI
- **自動化**: ワンクリックで翻訳完了、手動操作は不要

## システム要件

- Python 3.10以上
- PyTorch（CPU、CUDA、ROCm、XPU、MPSに対応）
- `.ttf`または`.otf`ファイルを含むフォントパック（ポータブルパッケージに同梱されています）
- 日本語のソーステキストにはLLM、その他の言語にはVLMが必要（APIまたはローカル実行）

## インストール

### ポータブルパッケージ（推奨）

リリースファイル一覧からスタンドアロンのzipファイルをダウンロードします：[Portable Build（ポータブルビルド）](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)

**システム要件：**

- **Windows:** PythonとGitが同梱されているため、事前準備は不要です。
- **Linux/macOS:** システムにPython 3.10以上およびGitがインストールされている必要があります。

> [!TIP]
> 新しいポータブルパッケージにデータを移行する必要がある場合：
>
> - `fonts`、`models`、`output` ディレクトリは、新しいポータブルパッケージにそのまま移動して再利用できます。
> - 同じ設定構成を維持したい場合は、`runtime` ディレクトリもそのままコピーできます。

### 手動インストール

1. リポジトリをクローンしてディレクトリに移動します。

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

2. 仮想環境を作成して有効化します（推奨）。

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. PyTorchをインストールします（参考：[PyTorchインストールガイド](https://pytorch.org/get-started/locally/)）

```bash
# 例 (CUDA 13.0)
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
# 例 (ROCm 7.1)
pip install torch==2.11.0+rocm7.1 torchvision==0.26.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1
# 例 (XPU)
pip install torch==2.11.0+xpu torchvision==0.26.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu
# 例 (MPS/CPU)
pip install torch==2.11.0 torchvision==0.26.0
```

4. Nunchakuのインストール（任意、Nunchakuバックエンド経由のFLUX.1 Kontext用）

- NunchakuのwheelファイルはPyPIにはありません。OSとPythonのバージョンに一致するものを、v1.3.0dev20260213のGitHubリリースURLから直接インストールしてください。CUDA専用で、RTX 2000シリーズ以降のグラフィックボードが必要です。

```bash
# 例 (Windows, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# 例 (Linux, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> sd.cpp/SDNQバックエンドを介してFluxモデルを使用する場合、Nunchakuは必須ではありません。

5. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## インストール後のセットアップ

### モデル（Models）

- アプリケーション起動時に、必要なモデルが自動的にダウンロードされロードされます。

### フォント（Fonts）

- `fonts/` の下にサブフォルダを作成し、その中に `.otf` または `.ttf` ファイルを配置します。
- ファイル名に `italic`（斜体）や `bold`（太字）が含まれていると、フォントのバリエーションが自動的に検出されやすくなります。
- フォルダ構造の例：

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

### LLMのセットアップ（LLM setup）

- プロバイダー：Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible（OpenAI互換）
- Web UI：Config（設定）タブからプロバイダー、モデル、APIキーを設定します（ローカルに保存されます）。
- CLI：コマンドライン引数または環境変数経由でキーやURLを渡します。
- 環境変数：`GOOGLE_API_KEY` / `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SPACEXAI_API_KEY` / `XAI_API_KEY`, `META_MODEL_API_KEY` / `META_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `MIMO_API_KEY`, `QWENCLOUD_API_KEY` / `QWEN_API_KEY`, `OPENCODE_API_KEY` / `OPENCODE_ZEN_API_KEY` / `OPENCODE_GO_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-Compatible（OpenAI互換）プロバイダーはローカルエンドポイント（例：`http://localhost:8080/v1`）およびAzure OpenAIエンドポイント（例：`https://<resource>.openai.azure.com`）をサポートしています

> [!NOTE]
> OpenAI互換プロバイダーを介して以下のモデルを使用する場合、自動的に検出されて最適化されたプロンプトが適用されます。これらはテキスト専用モデルであるため、2ステップ翻訳モードとローカルOCRモデルを有効にする必要があります。「特別指示 (Special Instructions)」フィールドは、対応する用語集/用語にマッピングされます（1行に1エントリ、例：`term -> translation`）。
>
> - **YanoljaNEXT-Rosetta** (例: `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`)
> - **Hy-MT2** (例: `tencent/Hy-MT2-7B`)。モデル推奨のサンプリングパラメータも自動的に設定されます。

### 吹き出し外（OSB）テキストのセットアップ（任意）

吹き出し外（書き文字など）のテキスト消去・再描画パイプラインを使用するには、以下のHugging Faceリポジトリへのアクセス権限を持つトークンが必要です：

- `deepghs/AnimeText_yolo`

#### トークン作成手順：

1. Hugging Faceアカウントにサインインするか、新規作成します。
2. 以下のページにアクセスし、利用規約に同意します：
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)（任意、Nunchakuバックエンド経由でFLUX.1 Kontextを使用する場合）
   - [SAM 3](https://huggingface.co/facebook/sam3)（SAM 3を使用する場合は任意）
3. Hugging Faceの設定で、ゲート付きリポジトリへの読み取り権限（"Read access to contents of public gated repos"）を持つ新しいAccess Tokenを作成します。
4. トークンをアプリケーションに追加します：
   - Web UI：Configタブの `hf_token` に入力します。
   - 環境変数（代替案）：`HF_TOKEN` を設定します。
5. セッション間でトークンを保持するため、設定を保存（Save Config）します。

## 実行

### Web UI（Gradio）

- **ポータブルパッケージ：**
  - `MangaTranslator/` ディレクトリ内で `start-webui.bat`（Windows）または `./start-webui.sh`（Linux/macOS）を実行
- **手動インストール：**
  - `python app.py --open-browser` を実行

起動オプションについては `python app.py --help` を実行してください。
初回の起動には1〜2分かかる場合があります。

起動後、ConfigタブでLLMプロバイダーを設定し、画像をアップロードして Translate をクリックします。

### CLI（コマンドライン）

使用例：

```bash
# 単一画像、日本語 → 英語、Googleプロバイダー、吹き出し外テキストパイプライン有効、吹き出し外テキスト専用フォント指定
python main.py --input <画像パス> \
  --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> \
  --osb-enable --osb-font-dir "fonts/Comicka"

# フォルダ内の一括処理、日本語 → 簡体字中国語、OpenAI互換プロバイダー（llama.cpp）、吹き出し外テキストパイプライン有効、吹き出し外テキスト専用フォント指定
python main.py --input <フォルダパス> --batch \
  --font-dir "fonts/Noto Sans SC" --output-language "Chinese (Simplified)" \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output --osb-enable --osb-font-dir "fonts/Noto Sans SC"

# クリーニング（消去）専用モード（翻訳なし）
python main.py --input <画像パス> --cleaning-only

# アップスケーリング専用モード（翻訳なし）
python main.py --input <画像パス> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# 全てのオプションを表示
python main.py --help
```

## ドキュメント

- [動作に必要な動作環境（ハードウェア）](HARDWARE_REQUIREMENTS.md)
- [おすすめのフォント](FONTS.md)
- [トラブルシューティング](TROUBLESHOOTING.md)

## アップデート

### ポータブルパッケージ

- ポータブルパッケージのルートから `update.bat`（Windows）または `./update.sh`（Linux/macOS）を実行

### 手動インストール

リポジトリのルートで実行します：

```bash
git pull
pip install -r requirements.txt  # 仮想環境を使用している場合は先にアクティベートしてください
```

## ライセンスとクレジット

- ライセンス：Apache-2.0（詳細は [LICENSE](../../../LICENSE) を参照）
- 開発者：[grinnch](https://github.com/meangrinch)

<details>
<summary><b>機械学習モデルおよびライブラリ</b></summary>

- YOLOv8m 吹き出し検出モデル: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 吹き出し検出モデル: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic テキスト・吹き出し検出 RT-DETR-v2: [ogkalu](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: Segment Anything in Images and Videos: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
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
