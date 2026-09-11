# CLI

一括バッチ処理、スクリプト実行、ヘッドレス翻訳を行うには `python main.py` を実行します。

## 使用例

```bash
# 単一画像、日本語 -> 英語、Googleプロバイダー、吹き出し外テキストパイプライン、カスタム吹き出し外フォント
python main.py --input <image_path> \
  --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> \
  --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>

# フォルダ一括バッチ処理、韓国語 -> 簡体字中国語、OpenAI互換プロバイダー (llama.cpp)、吹き出し外テキストパイプライン、カスタム吹き出し外フォント
python main.py --input <folder_path> --batch --font-dir "fonts/Noto Sans SC" \
  --input-language "Korean" --output-language "Chinese (Simplified)" \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output --osb-enable --osb-font-dir "fonts/Noto Sans SC" --osb-hf-token <...>

# 消去専用モード（翻訳なし）
python main.py --input <image_path> --cleaning-only

# アップスケーリング専用モード（翻訳なし）
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# すべてのオプションを表示
python main.py --help
```
