# CLI

Run `python main.py` for batch processing, scripts, and headless translation.

## Examples

```bash
# Single image, Japanese -> English, Google provider, OSB text pipeline, custom OSB text font
python main.py --input <image_path> \
  --font-dir "fonts/Komika Hand" --provider Google --google-api-key <...> \
  --osb-enable --osb-font-dir "fonts/Comicka" --osb-hf-token <...>

# Batch folder, Korean -> Chinese (Simplified), OpenAI-Compatible provider (llama.cpp), OSB text pipeline, custom OSB text font
python main.py --input <folder_path> --batch --font-dir "fonts/Noto Sans SC" \
  --input-language "Korean" --output-language "Chinese (Simplified)" \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output --osb-enable --osb-font-dir "fonts/Noto Sans SC" --osb-hf-token <...>

# Cleaning-only mode (no translation)
python main.py --input <image_path> --cleaning-only

# Upscaling-only mode (no translation)
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# Full options
python main.py --help
```
