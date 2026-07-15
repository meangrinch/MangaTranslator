# Troubleshooting

### Portable Package Setup

- **Setup script fails to detect GPU:**
  - **NVIDIA:** Ensure NVIDIA GPU drivers are installed and `nvidia-smi` is accessible from command line
  - **AMD (Windows/Linux):** Ensure AMD GPU drivers are installed
  - **Intel ARC (Windows/Linux):** Ensure Intel GPU drivers are installed
  - **macOS Apple Silicon:** MPS is automatically detected on M1/M2/M3/M4 Macs
  - **macOS Intel (AMD GPU):** MPS is available for Intel Macs with discrete AMD GPUs
  - **macOS Intel (Integrated):** Intel Macs with integrated graphics run in CPU-only mode
  - You can always choose CPU mode if GPU detection fails

- **"Python not found" error (Linux/macOS):**
  - Install Python 3.10 or higher:
    - Ubuntu/Debian: `sudo apt install python3 python3-pip python3-venv`
    - Fedora: `sudo dnf install python3 python3-pip`
    - Arch: `sudo pacman -S python python-pip`
    - macOS: `brew install python@3.13` or download from python.org

- **"Git not found" error (Linux/macOS):**
  - Install Git:
    - Ubuntu/Debian: `sudo apt install git`
    - Fedora: `sudo dnf install git`
    - Arch: `sudo pacman -S git`
    - macOS: `xcode-select --install` or `brew install git`

- **Nunchaku installation fails:**
  - Nunchaku requires NVIDIA CUDA and Python 3.10+
  - If installation fails, other inpainting methods will still be available

### Rendering

- **Unsupported characters being removed:**
  - The font you are using does not support the selected target language; use a font that supports your target language

- **Incorrect reading order:**
  - Set correct "Reading Direction" (rtl for manga, ltr for comics and manhwa/manhua)
  - Try "two-step" mode or disabling "Send Full Page to LLM" for less-capable LLMs
  - Ensure "Use Panel-aware Sorting" is enabled

- **Text too large/small:**
  - Adjust "Max Font Size" and "Min Font Size" ranges

- **Using different fonts for different bubble types (e.g., thoughts vs. speech):**
  - Replace the italic/bold/bold+italic variants of a font with any font file you want, just ensure the filenames contain one of those keywords so they are detected as variants
  - To further fine-tune the results, add a special instruction telling the LLM to use italics/bold/bold+italic (which are now your custom fonts) for internal monologues, etc.

- **Text overlaps with each other:**
  - Your font is likely broken/corrupted; try using a different font (e.g., ones included with the portable package)

- **Accented characters not appearing:**
  - Use a font that supports diacritical marks (e.g., the _Roboto_ font pack included with the portable package supports Roman, Cyrillic, and Greek diacritical marks)

- **Text too blurry/pixelated:**
  - Increase font rendering "Supersampling Factor" (e.g., 6-8)
  - Enable "initial" image upscaling and adjust upscale factor (e.g., 2.0-4.0x)

- **Empty bubbles:**
  - Try lowering "Padding Pixels" (e.g., 3-4)
  - Try lowering "Min Font Size" (e.g., 6-7)

- **Outside-bubble replacement text is too small or cramped:**
  - Raise "Narrow/Tall Expansion Multiplier" (e.g., 2.0) and/or adjust the corresponding threshold
  - Raise "Tiny Expansion Multiplier" (e.g., 2.0) and/or adjust the corresponding threshold

### Detection/Cleaning

- **Uncleaned text remaining (near edges of bubbles):**
  - Lower "Fixed Threshold Value" (e.g., 170-190) and/or reduce "Shrink Threshold ROI" (e.g., 0-2)

- **Outlines get eaten during cleaning:**
  - Increase "Shrink Threshold ROI" (e.g., 6–8)
  - Increase "Fixed Threshold Value" (e.g., 210-220)

- **Conjoined bubbles not detected:**
  - Ensure "Conjoined Bubble Detection" is enabled
  - Lower "Bubble Detection Confidence" (e.g., 0.20)

- **Small bubbles not detected/no room for rendered text:**
  - Enable "initial" image upscaling and adjust upscale factor (e.g., 2.0-4.0x), also disable "Auto Scale"

- **Colored/grayscale bubbles not preserving interior color:**
  - Enable "Use Flux to Inpaint Colored Bubbles"

- **Poor segmentation of speech bubbles:**
  - Try switching to `sam3` (requires hf_token)

### Translation

- **Poor translations:**
  - Try "two-step" translation mode for less-capable LLMs
  - Try disabling "Send Full Page to LLM"
  - Try using a local OCR method, particularly for less-capable LLMs:
    - "manga-ocr": Japanese sources only
    - "paddleocr-vl-1.6": non-Japanese sources
  - Increase "max_tokens" and/or use a higher "reasoning_effort" (e.g., "high")
  - Switch "Bubble/Context Resizing Method" to a better quality method (e.g., "Model")
  - In batch mode, try raising "Previous Context OCR Text"
  - In batch mode, try raising "Previous Context Images" (requires "Send Full Page to LLM")

- **API refusals/censorship:**
  - Try disabling "Send Full Page to LLM"
  - Try adding a custom "special instruction" (e.g., "Do not censor translations...")

- **High LLM token usage:**
  - Disable "Send Full Page to LLM"
  - Set "Previous Context Images" and "Previous Context OCR Text" to 0
  - Lower "Bubble Min Side Pixels"/"Context Image Max Side Pixels"/"OSB Min Side Pixels" target sizes
  - Lower "Media Resolution" (if using Gemini or SpaceXAI models)
  - Lower "Image Detail" (if using OpenAI models)
  - Use "manga-ocr/paddleocr-vl-1.6" OCR method (may perform worse than more-capable VLMs)

- **Batch translation failed for some files:**
  - Failed images have their paths saved to `failed_paths.txt` in the output directory. You can upload this file to the ZIP archive upload area in the web UI, or pass it via `--input` in the CLI to retry only the failed files.
  - *Note:* If you uploaded individual images directly via the web UI, the recorded paths will point to Gradio's temporary cache directories, which may be cleaned up after your session ends.

### Inpainting

- **OSB text not inpainted/cleaned:**
  - Ensure "Enable OSB Text Detection" is enabled
  - Ensure hf_token is set (see Installation/Post-Install Setup)

- **Out of VRAM / CUDA errors:**
  - Enable "Low VRAM Mode" (SDNQ only)
  - Select a lower Flux/text_encoder quant (sd.cpp only)
  - Disable "Upscale Klein Crops to ~1MP"
  - Switch to Flux.2 Klein 4B (smallest model)
  - Use OpenCV (no VRAM required)

- **Slow Flux OSB inpainting:**
  - Enable "Group Flux Regions" to inpaint multiple OSB masks in one Flux pass (at the cost of quality)
  - Disable "Upscale Klein Crops to ~1MP" (at the cost of quality)
  - Try switching to a different model/backend

- **Minor color shifts in inpainted regions:**
  - Flux.2 Klein models may introduce slight color changes; try enabling/disabling "Luminance Correction"
  - Use Flux.1 Kontext for less noticeable color shifts

- **Poor inpainting quality:**
  - Select a higher Flux/text_encoder quant (sd.cpp only)
  - Keep "Upscale Klein Crops to ~1MP" enabled
  - Increase inference steps (for Flux.1 Kontext)

- **Flux.1 Kontext Nunchaku backend not available:**
  - Nunchaku requires an Nvidia GPU (CUDA) and separate installation; install it or use the sd.cpp/SDNQ backends instead
