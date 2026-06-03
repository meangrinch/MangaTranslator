# トラブルシューティング

### ポータブル版のセットアップ (Portable Package Setup)

- **セットアップスクリプトが GPU を検出できない:**
  - **NVIDIA:** NVIDIA GPU ドライバがインストールされており、コマンドラインから `nvidia-smi` にアクセスできることを確認してください。
  - **AMD (Windows/Linux):** AMD GPU ドライバがインストールされていることを確認してください。
  - **Intel ARC (Windows/Linux):** Intel GPU ドライバがインストールされていることを確認してください。
  - **macOS Apple Silicon:** M1/M2/M3/M4 Mac では MPS が自動的に検出されます。
  - **macOS Intel (AMD 独立 GPU):** 独立した AMD GPU を搭載した Intel Mac では MPS が利用可能です。
  - **macOS Intel (統合グラフィックス):** 統合グラフィックスを搭載した Intel Mac は CPU のみモードで動作します。
  - GPU 検出に失敗した場合は、いつでも CPU モードを選択して実行できます。

- **「Python not found」（Python が見つかりません）エラー (Linux/macOS):**
  - Python 3.10 以上をインストールしてください:
    - Ubuntu/Debian: `sudo apt install python3 python3-pip python3-venv`
    - Fedora: `sudo dnf install python3 python3-pip`
    - Arch: `sudo pacman -S python python-pip`
    - macOS: `brew install python@3.13` または python.org からダウンロード

- **「Git not found」（Git が見つかりません）エラー (Linux/macOS):**
  - Git をインストールしてください:
    - Ubuntu/Debian: `sudo apt install git`
    - Fedora: `sudo dnf install git`
    - Arch: `sudo pacman -S git`
    - macOS: `xcode-select --install` または `brew install git`

- **Nunchaku のインストールが失敗する:**
  - Nunchaku には NVIDIA CUDA と Python 3.10+ が必要です。
  - インストールに失敗した場合でも、他のインペインティング方法を利用可能です。

### レンダリング (Rendering)

- **サポートされていない文字が削除される:**
  - 使用しているフォントが選択したターゲット言語をサポートしていません。ターゲット言語をサポートするフォントを使用してください。

- **読む順番が正しくない:**
  - 正しい「読む方向（Reading Direction）」を設定してください（日本の漫画の場合は `rtl` (右から左)、アメコミなどの場合は `ltr` (左から右)）。
  - 性能の低い LLM の場合は、「2ステップ（two-step）」モードを試すか、「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてみてください。
  - 「コマを考慮したソート（Use Panel-aware Sorting）」が有効になっていることを確認してください。

- **テキストが大きすぎる / 小さすぎる:**
  - 「最大フォントサイズ（Max Font Size）」と「最小フォントサイズ（Min Font Size）」の範囲を調整してください。

- **吹き出しのタイプ（例: 心の声 vs 通常のセリフ）に応じて異なるフォントを使用する:**
  - フォントの斜体（italic）/太字（bold）/太斜体（bold+italic）のバリアントをご希望のフォントファイルに置き換えてください。その際、プログラムがバリアントとして検出できるように、ファイル名にそれらのキーワードが含まれていることを確認してください。
  - さらに細かく調整するには、心の声などに斜体/太字/太斜体（置き換えたカスタムフォント）を使用するよう LLM に指示する「特別指示（special instruction）」を追加してください。

- **テキスト同士が重なり合う:**
  - フォントファイルが破損している可能性があります。別のフォントを試してください（ポータブル版に含まれているフォントなど）。

- **アクセント付き文字が表示されない:**
  - ダイアクリティカルマーク（発音区別符号）をサポートするフォントを使用してください（ポータブル版に同梱されている _Roboto_ フォントパックは、ラテン、キリル、ギリシャのダイアクリティカルマークをサポートしています）。

- **テキストがぼやけている / ピクセル化している:**
  - フォントレンダリングの「スーパーサンプリング係数（Supersampling Factor）」を上げてください（例: 6-8）。
  - 「初期（initial）」画像アップスケーリングを有効にし、アップスケール倍率を調整してください（例: 2.0x-4.0x）。

- **空白の吹き出しができる:**
  - 「パディングピクセル（Padding Pixels）」を下げてみてください（例: 3-4）。
  - 「最小フォントサイズ（Min Font Size）」を下げてみてください（例: 6-7）。

- **吹き出し外の置き換えテキストが小さすぎる、または窮屈すぎる:**
  - 「幅狭・縦長領域の拡大倍率（Narrow/Tall Expansion Multiplier）」および/または対応するしきい値を上げてください（例: 2.0）。
  - 「極小領域の拡大倍率（Tiny Expansion Multiplier）」および/または対応するしきい値を上げてください（例: 2.0）。

### 検出 / クリーニング (Detection/Cleaning)

- **消去しきれなかったテキストが残る（吹き出しの端の近くなど）:**
  - 「固定しきい値（Fixed Threshold Value）」を下げる（例: 170-190）か、「しきい値 ROI 縮小（Shrink Threshold ROI）」を減らしてください（例: 0-2）。

- **クリーニング中に文字の輪郭が削れてしまう:**
  - 「しきい値 ROI 縮小（Shrink Threshold ROI）」を増やす（例: 6–8）か、「固定しきい値（Fixed Threshold Value）」を上げてください（例: 210-220）。

- **連結された吹き出しが検出されない:**
  - 「連結吹き出し検出（Conjoined Bubble Detection）」が有効になっていることを確認してください。
  - 「吹き出し検出의 信頼度しきい値（Bubble Detection Confidence）」を下げてみてください（例: 0.20）。

- **小さな吹き出しが検出されない / レンダリングされたテキストを配置するスペースがない:**
  - 「初期（initial）」画像アップスケーリングを有効にしてアップスケール倍率を調整（例: 2.0x-4.0x）し、「自動スケール（Auto Scale）」を無効にしてください。

- **カラー/グレースケールの吹き出しで、内部の色が保持されない:**
  - 「カラー吹き出しに Flux インペイントを使用（Use Flux to Inpaint Colored Bubbles）」を有効にしてください。

- **吹き出しのセグメンテーション品質が低い:**
  - `sam3` に切り替えてみてください（`hf_token` が必要です）。

### 翻訳 (Translation)

- **翻訳の品質が悪い:**
  - 性能の低い LLM の場合は、「2ステップ（two-step）」翻訳モードを試してください。
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてみてください。
  - 性能 of 低い LLM を使用する場合は、ローカルの OCR 方法を試してください:
    - "manga-ocr": 日本語の原文のみ
    - "paddleocr-vl": 日本語以外の原文
  - `max_tokens` を増やすか、より高い「推理力（reasoning_effort）」（例: "high"）を使用してください。
  - 「吹き出し/文脈リサイズ方法（Bubble/Context Resizing Method）」をより高品質な方法（例: "Model"）に切り替えてください。
  - バッチモードでは、「前文の OCR テキスト（Previous Context OCR Text）」を増やしてみてください。
  - バッチモードでは、「前文の画像（Previous Context Images）」を増やしてみてください（「ページ全体を LLM に送信」が必要です）。

- **API の拒否 / 検閲による規制:**
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてみてください。
  - カスタムの「特別指示（special instruction）」を追加してみてください（例: 「翻訳を検閲しないでください...」）。

- **LLM のトークン使用量が高すぎる:**
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてください。
  - 「前文の画像（Previous Context Images）」および「前文の OCR テキスト（Previous Context OCR Text）」を 0 に設定してください。
  - 「吹き出し最小短辺ピクセル（Bubble Min Side Pixels）」/「文脈画像最大長辺ピクセル（Context Image Max Side Pixels）」/「OSB 最小短辺ピクセル（OSB Min Side Pixels）」のターゲットサイズを下げてください。
  - 「メディア解像度（Media Resolution）」を下げてください（Gemini または xAI 모델を使用している場合）。
  - 「画像ディテール（Image Detail）」を下げてください（OpenAI モデルを使用している場合）。
  - "manga-ocr/paddleocr-vl" OCR 方法を使用してください（より高性能な VLM を直接使用するよりも性能が劣る場合があります）。

### インペインティング (Inpainting)

- **吹き出し外（OSB）のテキストがインペイント / クリーニングされない:**
  - 「OSB テキスト検出の有効化（Enable OSB Text Detection）」が有効になっていることを確認してください。
  - `hf_token` が設定されていることを確認してください（セットアップ/セットアップ後の設定を参照）。

- **VRAM 不足 / CUDA エラー:**
  - 「低 VRAM モード（Low VRAM Mode）」を有効にしてください（SDNQ のみ）。
  - より低い Flux/text_encoder 量子化を選択してください（sd.cpp のみ）。
  - 「Klein クロップ領域を ~1MP にアップスケール（Upscale Klein Crops to ~1MP）」を無効にしてください。
  - Flux.2 Klein 4B（最も小さいモデル）に切り替えてください。
  - OpenCV を使用してください（VRAM 不要）。

- **Flux OSB インペイントが遅い:**
  - 「Flux 領域のグループ化（Group Flux Regions）」を有効にして、1回の Flux パスで複数の OSB マスクをインペイントします（品質が低下する場合があります）。
  - 「Klein クロップ領域を ~1MP にアップスケール（Upscale Klein Crops to ~1MP）」を無効にしてください（品質が低下する場合があります）。
  - 別のモデル/バックエンドに切り替えてみてください。

- **インペイント領域のわずかな色ズレ:**
  - Flux.2 Klein モデルはわずかな色の変化を導入する場合があります。「輝度補正（Luminance Correction）」の有効/無効を切り替えてみてください。
  - 色ズレを目立たなくするために、Flux.1 Kontext を使用してください。

- **インペイントの品質が低い:**
  - より高い Flux/text_encoder 量子化を選択してください（sd.cpp のみ）。
  - 「Klein クロップ領域を ~1MP にアップスケール（Upscale Klein Crops to ~1MP）」を有効のままにしてください。
  - 推論ステップ（inference steps）数を増やしてください（Flux.1 Kontext の場合）。

- **Flux.1 Kontext Nunchaku バックエンドが利用できない:**
  - Nunchaku には Nvidia GPU (CUDA) と個別のインストールが必要です。インストールするか、代わりに sd.cpp/SDNQ バックエンドを使用してください。
