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
  - Python 3.11 以上をインストールしてください:
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
  - Nunchaku には NVIDIA CUDA、Python 3.11+、PyTorch 2.12 以下が必要です。
  - インストールに失敗した場合でも、他のインペインティング方法を利用可能です。

### レンダリング (Rendering)

- **サポートされていない文字が削除される:**
  - 使用しているフォントが選択したターゲット言語をサポートしていません。ターゲット言語をサポートするフォントを使用してください。

- **読む順番が正しくない:**
  - 正しい「読む方向（Reading Direction）」を設定してください（日本の漫画の場合は `rtl` (右から左)、アメコミなどの場合は `ltr` (左から右)）。
  - 性能の低い LLM の場合は、「2ステップ（two-step）」モードを試してください。
  - 「コマを考慮したソート（Use Panel-aware Sorting）」が有効になっていることを確認してください。

- **テキストが大きすぎる / 小さすぎる:**
  - 「最大フォントサイズ（Max Font Size）」と「最小フォントサイズ（Min Font Size）」の範囲を調整してください（「自動フォントサイズ（Auto Font Size）」はこの範囲内で動作します）。
  - 元のテキストサイズに合わせるのではなく、各吹き出しに収まる最大サイズを使用するには、「自動フォントサイズ（Auto Font Size）」を無効にしてください。

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

- **OSB テキストが小さすぎる、または窮屈すぎる:**
  - 「幅狭・縦長領域の拡大倍率（Narrow/Tall Expansion Multiplier）」および/または対応するしきい値を上げてください（例: 2.0）。
  - 「極小領域の拡大倍率（Tiny Expansion Multiplier）」および/または対応するしきい値を上げてください（例: 2.0）。

- **OSB テキストの傾き（回転）が不自然、または無効化したい場合:**
  - 「元のテキストの傾きに合わせる（Match Original Text Tilt）」を無効にしてください

### 検出 / クリーニング (Detection/Cleaning)

- **消去しきれなかったテキストが残る（吹き出しの端の近くなど）:**
  - 「固定しきい値（Fixed Threshold Value）」を下げる（例: 170-190）か、「しきい値 ROI 縮小（Shrink Threshold ROI）」を減らしてください（例: 0-2）。

- **クリーニング中に文字の輪郭が削れてしまう:**
  - 「しきい値 ROI 縮小（Shrink Threshold ROI）」を増やす（例: 6–8）か、「固定しきい値（Fixed Threshold Value）」を上げてください（例: 210-220）。

- **連結された吹き出しが検出されない:**
  - 「連結吹き出し検出（Conjoined Bubble Detection）」が有効になっていることを確認してください。
  - 「吹き出し信頼度しきい値（Bubble Confidence Threshold）」を下げてみてください（例: 0.20）。

- **小さな吹き出しが検出されない / レンダリングされたテキストを配置するスペースがない:**
  - 「初期（initial）」画像アップスケーリングを有効にしてアップスケール倍率を調整（例: 2.0x-4.0x）し、「自動スケール（Auto-Scale to Image Size）」を無効にしてください。

- **カラー/グレースケールの吹き出しで、内部の色が保持されない:**
  - 「OSB テキスト検出の有効化（Enable OSB Text Detection）」を有効にしてください。カラー吹き出しはその OSB インペインティングモデルでクリーニングされます。
  - 「カラー吹き出しにインペインティングモデルを使用（Use Inpainting Model for Colored Bubbles）」が有効になっていることを確認してください。

- **吹き出しのセグメンテーション品質が低い:**
  - `sam3` に切り替えてみてください（`hf_token` が必要です）。

- **OSB テキストが検出されない:**
  - 「OSB テキスト検出の信頼度（OSB Text Detection Confidence）」を下げます。

- **作画が消される、またはテキストでないものが OSB テキストとして翻訳される:**
  - 「見逃したテキストを回収（Recover Missed Text）」を無効にします。

### 翻訳 (Translation)

- **翻訳の品質が悪い:**
  - 性能の低い LLM の場合は、「2ステップ（two-step）」翻訳モードを試してください。
  - 「2ステップ（two-step）」モードでは、「OCR 補正（OCR Correction）」を有効にして、文脈に基づいて不完全な OCR 結果を補正してみてください。
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」は有効のままにしてください。ページがないと、効果音が何の音か分からないまま翻訳されます。
  - 性能 of 低い LLM を使用する場合は、ローカルの OCR 方法を試してください:
    - "manga-ocr": 日本語の原文のみ
    - "paddleocr-vl-1.6": 日本語以外の原文
  - `max_tokens` を増やすか、より高い「推理力（reasoning_effort）」（例: "high"）を使用してください。
  - 「吹き出し/文脈リサイズ方法（Bubble/Context Resizing Method）」をより高品質な方法（例: "Model"）に切り替えてください。
  - バッチモードでは、「前文の OCR テキスト（Previous Context OCR Text）」を増やしてみてください。
  - バッチモードでは、「前文の画像（Previous Context Images）」を増やしてみてください（「ページ全体を LLM に送信」が必要です）。
  - バッチモードでは、「用語集ファイル（Glossary File）」に前の章の用語集を指定すると、章をまたいで名前の訳を統一できます（新しい用語集は `output/.glossary` に保存されます）。

- **ローカル LLM で空の応答・タイムアウト・リクエストの遅延が発生する:**
  - 考えすぎを防ぐため、サーバー側で推論（reasoning）を無効にしてください（例: llama.cpp の `--reasoning off`）。

- **API の拒否 / 検閲による規制:**
  - 「エロ同人誌モード（Ero-doujinshi Mode）」を有効にしてみてください（18禁・成人向けコンテンツの場合）。
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてみてください。
  - カスタムの「特別指示（special instruction）」を追加してみてください（例: 「翻訳を検閲しないでください...」）。

- **LLM のトークン使用量が高すぎる:**
  - 「ページ全体を LLM に送信（Send Full Page to LLM）」を無効にしてください。
  - 「前文の画像（Previous Context Images）」および「前文の OCR テキスト（Previous Context OCR Text）」を 0 に設定してください。
  - 「吹き出し最小短辺ピクセル（Bubble Min Side Pixels）」/「文脈画像最大長辺ピクセル（Context Image Max Side Pixels）」/「OSB 最小短辺ピクセル（OSB Text Min Side Pixels）」のターゲットサイズを下げてください（画像 1 枚あたりのトークン数が固定のモデルでは効果がありません）。
  - 「メディア解像度（Media Resolution）」を下げてください（Gemini または SpaceXAI モデルを使用している場合）。
  - 「画像ディテール（Image Detail）」を下げてください（OpenAI モデルを使用している場合）。
  - "manga-ocr/paddleocr-vl-1.6" OCR 方法を使用してください（より高性能な VLM を直接使用するよりも性能が劣る場合があります）。
  - 「翻訳キャッシュの強制（Force Cache Translations）」を有効にして、非決定論的サンプリングでの再実行時にもキャッシュされた翻訳を再利用してください。

- **一部のファイルのバッチ翻訳に失敗した:**
  - 翻訳に失敗した画像は出力フォルダ内の `failed` フォルダにコピーされ、その隣の `failed_paths.txt` に一覧が保存されます。WebUI の ZIP アップロードエリアにこのファイルをアップロードするか、CLI で `--input` 引数に指定することで、失敗したファイルのみを再試行できます。再試行の結果は元の出力フォルダに同じフォルダ構成で保存され（CLI では `--output` で上書き可能）、その実行の用語メモリファイルが引き続き使用されます。
  - 吹き出しが1つでも空のまま残ったページ（有効な翻訳がない、またはテキストを描画できなかった場合）も失敗として扱われます。途中までの翻訳結果は出力フォルダに保存され、そのページを再試行すると上書きされます。
  - 再試行で翻訳に成功した画像のコピーは `failed` フォルダから削除され、`failed_paths.txt` はまだ失敗しているファイルのみを列挙するよう書き直されます。
  - *注意:* 一覧と一緒に WebUI のアップローダーへ画像を追加した場合、それらは新しいバッチとして処理され、新しい出力フォルダに保存されます。

### インペインティング (Inpainting)

- **吹き出し外（OSB）のテキストがインペイント / クリーニングされない:**
  - 「OSB テキスト検出の有効化（Enable OSB Text Detection）」が有効になっていることを確認してください。
  - `hf_token` が設定されていることを確認してください（設定/吹き出し外テキスト設定を参照）。

- **VRAM 不足 / CUDA エラー:**
  - より低い拡散モデル/テキストエンコーダーの量子化を選択してください（sd.cpp のみ）。
  - 「クロップ領域を ~1MP にアップスケール（Upscale Crops to ~1MP）」を無効にしてください。
  - Flux.2 Klein 4B（最も小さい拡散モデル）または LaMa Manga（VRAM 1 GB 未満）に切り替えてください。
  - OpenCV を使用してください（VRAM 不要）。

- **OSB インペイントが遅い:**
  - 十分な VRAM がある場合は「低 VRAM モード（Low VRAM Mode）」を無効にしてください（SDNQ のみ）。
  - 「インペインティング領域のグループ化（Group Inpainting Regions）」を有効にして、1回のパスで複数の OSB マスクをインペイントします（品質が低下する場合があります）。
  - 「クロップ領域を ~1MP にアップスケール（Upscale Crops to ~1MP）」を無効にしてください（品質が低下する場合があります）。
  - 別のモデル/バックエンドに切り替えてみてください。LaMa Manga ははるかに高速ですが、再現される細部は少なくなります。
  - 「テキストセグメンテーションモデルを使用（Use Text Segmentation Model）」を無効にして、SAM-TS-L テキストセグメンテーションモデルを省略してください。

- **インペイント領域のわずかな色ズレ:**
  - Flux.2 Klein モデルはわずかな色の変化を導入する場合があります。「輝度補正（Luminance Correction）」の有効/無効を切り替えてみてください。
  - 色ズレを目立たなくするために、Flux.1 Kontext を使用してください。

- **インペイントの品質が低い:**
  - より高い拡散モデル/テキストエンコーダーの量子化を選択してください（sd.cpp のみ）。
  - 「クロップ領域を ~1MP にアップスケール（Upscale Crops to ~1MP）」を有効のままにしてください。
  - 推論ステップ（inference steps）数を増やしてください（Flux.1 Kontext の場合）。

- **インペイント後に OSB 文字の断片が残る:**
  - 「周囲の作画を保持（Preserve Surrounding Art）」を無効にすると、グリフ単位のマスクに限定せず、テキストボックス全体を貼り付けます。

- **Flux.1 Kontext Nunchaku バックエンドが利用できない:**
  - Nunchaku には Nvidia GPU (CUDA) と個別のインストールが必要です。インストールするか、代わりに sd.cpp/SDNQ バックエンドを使用してください。
