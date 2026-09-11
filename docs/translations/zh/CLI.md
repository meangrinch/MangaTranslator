# CLI

运行 `python main.py` 进行批量处理、脚本调用和无界面翻译。

## 示例

```bash
# 单图处理，日译英，Google 服务商，对话框外文本管线，自定义对话框外文本字体
python main.py --input <image_path> \
  --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> \
  --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>

# 批量文件夹处理，韩译简体中文，OpenAI 兼容服务商 (llama.cpp)，对话框外文本管线，自定义对话框外文本字体
python main.py --input <folder_path> --batch --font-dir "fonts/Noto Sans SC" \
  --input-language "Korean" --output-language "Chinese (Simplified)" \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output --osb-enable --osb-font-dir "fonts/Noto Sans SC" --osb-hf-token <...>

# 仅擦除模式（不翻译）
python main.py --input <image_path> --cleaning-only

# 仅超分辨率模式（不翻译）
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# 查看完整参数
python main.py --help
```
