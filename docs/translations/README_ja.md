[English](../../README.md) | [简体中文](README_zh.md) | [한국어](README_ko.md) | [日本語](README_ja.md)

## MangaTranslator

AIを使用して漫画やコミックページの画像翻訳を自動化する、GradioベースのWebアプリケーションです。吹き出し内のテキストと吹き出し外のテキストの両方を翻訳対象とします。59言語およびカスタムフォントパックの使用に対応しています。

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">オリジナル</th>
      <th style="text-align: left">翻訳後（ワンクリックで完了）</th>
    </tr>
    <tr>
      <td><img src="../images/example_original.jpg" width="400" /></td>
      <td><img src="../images/example_translation.jpg" width="400" /></td>
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

- **検出**: 吹き出しの検出とセグメンテーション（YOLO + SAM 2.1/3）
- **クリーニング**: 吹き出し内および吹き出し外（OSB）テキストのインペインティング（Flux.2 Klein、Flux.1 Kontext、またはOpenCV）
- **翻訳**: LLMを活用したOCRと翻訳（59言語対応）
- **レンダリング**: 位置合わせやカスタムフォントパックをサポートしたテキスト描画
- **アップスケーリング**: 画質向上のための2x-AnimeSharpV4対応
- **処理**: ディレクトリ構造を維持した、単一画像またはバッチ処理（ZIP対応）
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

**セットアップ方法：**

1. zipファイルを解凍します。
2. プラットフォームに応じたセットアップスクリプトを実行します：
   - **Windows:** `setup.bat` をダブルクリック
   - **Linux/macOS:** ターミナルで `./setup.sh` を実行
3. システム環境に合わせて、PyTorchが自動的に検出およびインストールされます。
4. 生成されたランチャースクリプトを `./MangaTranslator/` フォルダ内から起動します：
   - **Windows:** `start-webui.bat` を実行
   - **Linux/macOS:** `start-webui.sh` を実行

同梱フォントパック：

- _Komika_（通常のセリフ用）
- _Cookies_（吹き出し外の書き文字用）
- _Comicka_（両方に対応）
- _Roboto_（アクセント記号対応）
- _Noto Sans SC_（簡体字中国語対応）

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
pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
# 例 (ROCm 7.1)
pip install torch==2.10.0+rocm7.1 torchvision==0.25.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1
# 例 (XPU)
pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu
# 例 (MPS/CPU)
pip install torch==2.10.0 torchvision==0.25.0
```

4. Nunchakuのインストール（任意、Flux.1 Kontext Nunchakuバックエンド用）

- NunchakuのwheelファイルはPyPIにはありません。OSとPythonのバージョンに一致するものを、v1.2.1のGitHubリリースURLから直接インストールしてください。CUDA専用で、RTX 2000シリーズ以降のグラフィックボードが必要です。

```bash
# 例 (Windows, Python 3.13, PyTorch 2.10.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp313-cp313-win_amd64.whl
```

> [!NOTE]
> SDNQバックエンドを介してFluxモデルを使用する場合、Nunchakuは必須ではありません。

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
└─ Komika/
   ├─ KOMIKA-HAND.ttf
   └─ KOMIKA-HANDBOLD.ttf
```

### LLMのセットアップ（LLM setup）

- プロバイダー：Google, OpenAI, Anthropic, xAI, DeepSeek, Z.ai, Moonshot AI, OpenRouter, OpenAI-Compatible（OpenAI互換）
- Web UI：Config（設定）タブからプロバイダー、モデル、APIキーを設定します（ローカルに保存されます）。
- CLI：コマンドライン引数または環境変数経由でキーやURLを渡します。
- 環境変数：`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI互換のデフォルトURL：`http://localhost:8080/v1`

> [!NOTE]
> OpenAI互換プロバイダーを介してYanoljaNEXT-Rosettaモデル（例：`yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`）を使用する場合、自動的に検出されて最適化されたプロンプトが適用されます。これらはテキスト専用モデルであるため、2ステップ翻訳モードとローカルOCRモデルを有効にする必要があります。「特別指示 (Special Instructions)」フィールドはRosettaの翻訳用語集にマッピングされます（1行に1エントリ、例：`Yanolja NEXT -> 야놀자넥스트`）。

### 吹き出し外（OSB）テキストのセットアップ（任意）

吹き出し外（書き文字など）のテキスト消去・再描画パイプラインを使用するには、以下のHugging Faceリポジトリへのアクセス権限を持つトークンが必要です：

- `deepghs/AnimeText_yolo`

#### トークン作成手順：

1. Hugging Faceアカウントにサインインするか、新規作成します。
2. 以下のページにアクセスし、利用規約に同意します：
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)（NunchakuでKontextを使用する場合は必須）
   - [SAM 3](https://huggingface.co/facebook/sam3)（SAM 3を使用する場合は任意）
3. Hugging Faceの設定で、ゲート付きリポジトリへの読み取り権限（"Read access to contents of public gated repos"）を持つ新しいAccess Tokenを作成します。
4. トークンをアプリケーションに追加します：
   - Web UI：Configタブの `hf_token` に入力します。
   - 環境変数（代替案）：`HUGGINGFACE_TOKEN` を設定します。
5. セッション間でトークンを保持するため、設定を保存（Save Config）します。

## 実行

### Web UI（Gradio）

- **ポータブルパッケージ：**
  - Windows：`MangaTranslator`フォルダ内の `start-webui.bat` をダブルクリック
  - Linux/macOS：ターミナルで `MangaTranslator`フォルダ内の `./start-webui.sh` を実行
- **手動インストール：**
  - Windows/Linux/macOS：`python app.py --open-browser` を実行

コマンドライン引数：`--models`（デフォルトは `./models`）、`--fonts`（デフォルトは `./fonts`）、`--port`（デフォルトは `7676`）、`--cpu`。
初回の起動には1〜2分かかる場合があります。

起動後、ConfigタブでLLMプロバイダーを設定し、画像をアップロードして Translate をクリックします。

### CLI（コマンドライン）

使用例：

```bash
# 単一画像、日本語 → 英語、Googleプロバイダー使用
python main.py --input <画像パス> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AIキー...>

# フォルダ内の一括処理、翻訳言語の指定、OpenAI互換プロバイダー (llama.cpp) 使用
python main.py --input <フォルダパス> --batch \
  --font-dir "fonts/Komika" \
  --input-language <翻訳元言語> --output-language <翻訳先言語> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output

# 単一画像、日本語 → 英語 (Google)、吹き出し外テキストパイプライン有効、吹き出し外テキスト専用フォント指定
python main.py --input <画像パス> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AIキー...> \
  --osb-enable --osb-font-dir "fonts/Clementine"

# クリーニング（消去）専用モード（翻訳や描画は行わず、文字消去のみを実行）
python main.py --input <画像パス> --cleaning-only

# アップスケーリング専用モード（検出や翻訳は行わず、画像のみを2倍に拡大）
python main.py --input <画像パス> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# テストモード（実際の翻訳は行わず、プレースホルダーテキストを描画）
python main.py --input <画像パス> --test-mode

# 全てのオプションを表示
python main.py --help
```

## ドキュメント

- [動作に必要な動作環境（ハードウェア）](../HARDWARE_REQUIREMENTS.md)
- [おすすめのフォント](../FONTS.md)
- [トラブルシューティング](../TROUBLESHOOTING.md)

## アップデート

### ポータブルパッケージ

- Windows：ポータブルパッケージのルートにある `update.bat` を実行
- Linux/macOS：ポータブルパッケージのルートにある `./update.sh` を実行

### 手動インストール

リポジトリのルートで実行します：

```bash
git pull
pip install -r requirements.txt  # 仮想環境を使用している場合は先にアクティベートしてください
```

## ライセンスとクレジット

- ライセンス：Apache-2.0（詳細は [LICENSE](../../LICENSE) を参照）
- 開発者：[grinnch](https://github.com/meangrinch)

<details>
<summary><b>機械学習モデルおよびライブラリ</b></summary>

- YOLOv8m 吹き出し検出モデル: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 吹き出し検出モデル: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic 吹き出し検出モデル YOLOv8m: [ogkalu](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
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
