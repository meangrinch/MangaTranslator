# Troubleshooting

### Layout & Text Rendering

- **Incorrect reading order:**
  - Set correct "Reading Direction" (rtl for manga, ltr for comics)
  - Try "two-step" mode or disabling "Send Full Page to LLM" for less-capable LLMs
  - Ensure "Use Panel-aware Sorting" is enabled
- **Text too large/small:**
  - Adjust "Max Font Size" and "Min Font Size" ranges
- **Text overlaps with each other:**
  - Your font is likely broken/corrupted; try using a different font (e.g., ones included with the portable package)
- **Text too blurry/pixelated:**
  - Increase font rendering "Supersampling Factor" (e.g., 6-8)
  - Enable "initial" image upscaling and adjust upscale factor (e.g., 2.0-4.0x)

### Bubble Detection & Cleaning

- **Uncleaned text remaining (near edges of bubbles):**
  - Lower "Fixed Threshold Value" (e.g., 180) and/or reduce "Shrink Threshold ROI" (e.g., 0–2)
- **Outlines get eaten during cleaning:**
  - Increase "Shrink Threshold ROI" (e.g., 6–8)
- **Conjoined bubbles not detected:**
  - Ensure "Detect Conjoined Bubbles" is enabled
  - Lower "Bubble Detection Confidence" (e.g., 0.20)
- **Small bubbles not detected/no room for rendered text:**
  - Enable "initial" image upscaling and adjust upscale factor (e.g., 2.0-4.0x), also disable "Auto Scale"
- **Colored/complex bubbles not preserving interior color:**
  - Enable "Use Flux Kontext to Inpaint Colored Bubbles" (requires Nunchaku/hf_token)

### Translation, LLM & API Issues

- **Poor translations/API refusals:**
  - Try "two-step" translation mode for less-capable LLMs
  - Try disabling "Send Full Page to LLM"
  - Try using "manga-ocr" OCR method, particularly for less-capable LLMs (Japanese sources only)
  - Increase "max_tokens" and/or use a higher "reasoning_effort" (e.g., "high")
  - Switch "Bubble/Context Resizing Method" to a better quality method (e.g., "Model")
- **High LLM token usage:**
  - Disable "Send Full Page to LLM"
  - Lower "Bubble Min Side Pixels"/"Context Image Max Side Pixels"/"OSB Min Side Pixels" target sizes
  - Lower "Media Resolution" (if using Gemini models)
  - Use "manga-ocr" OCR method (Japanese sources only; may perform worse than more-capable VLMs)

### Inpainting & Performance (Flux/OSB)

- **OSB text not inpainted/cleaned:**
  - Ensure "Enable OSB Text Detection" is enabled
  - Ensure hf_token is set (see Installation/Post-Install Setup)
  - Ensure Nunchaku is installed (if using Flux)
- **Flux Kontext too heavy/slow or OSB text hard to read:**
  - Enable "Force OpenCV Inpainting Instead of Flux"
