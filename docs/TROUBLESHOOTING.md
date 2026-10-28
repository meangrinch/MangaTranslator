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
  - Install Python 3.11 or higher:
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
  - Nunchaku requires NVIDIA CUDA, Python 3.11+, and PyTorch ≤2.12
  - If installation fails, other inpainting methods will still be available

### Rendering

- **Unsupported characters being removed:**
  - The font you are using does not support the selected target language; use a font that supports your target language

- **Incorrect reading order:**
  - Set correct "Reading Direction" (rtl for manga, ltr for comics and manhwa/manhua)
  - Try "two-step" mode for less-capable LLMs
  - Ensure "Use Panel-aware Sorting" is enabled

- **Text too large/small:**
  - Adjust "Max Font Size" and "Min Font Size" ranges ("Auto Font Size" stays within them)
  - Disable "Auto Font Size" to use the largest size that fits each bubble instead of matching the original text size

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

- **OSB text is too small or cramped:**
  - Raise "Narrow/Tall Expansion Multiplier" (e.g., 2.0) and/or adjust the corresponding threshold
  - Raise "Tiny Expansion Multiplier" (e.g., 2.0) and/or adjust the corresponding threshold

- **OSB text tilt/rotation looks incorrect or is unwanted:**
  - Disable "Match Original Text Tilt"

### Detection/Cleaning

- **Uncleaned text remaining (near edges of bubbles):**
  - Lower "Fixed Threshold Value" (e.g., 170-190) and/or reduce "Shrink Threshold ROI" (e.g., 0-2)

- **Outlines get eaten during cleaning:**
  - Increase "Shrink Threshold ROI" (e.g., 6–8)
  - Increase "Fixed Threshold Value" (e.g., 210-220)

- **Conjoined bubbles not detected:**
  - Ensure "Conjoined Bubble Detection" is enabled
  - Lower "Bubble Confidence Threshold" (e.g., 0.20)

- **Small bubbles not detected/no room for rendered text:**
  - Enable "initial" image upscaling and adjust upscale factor (e.g., 2.0-4.0x), also disable "Auto-Scale to Image Size"

- **Colored/grayscale bubbles not preserving interior color:**
  - Enable "Enable OSB Text Detection"; colored bubbles are cleaned with its inpainting model
  - Ensure "Use Inpainting Model for Colored Bubbles" is enabled

- **Poor segmentation of speech bubbles:**
  - Try switching to `sam3` (requires hf_token)

- **OSB text not detected:**
  - Lower "OSB Text Detection Confidence"

- **Art erased or non-text translated as OSB text:**
  - Disable "Recover Missed Text"

### Translation

- **Poor translations:**
  - Try "two-step" translation mode for less-capable LLMs
  - In "two-step" mode, try enabling "OCR Correction" to contextually repair poor OCR transcripts
  - Keep "Send Full Page to LLM" enabled; without the page, sound effects are translated without seeing what they belong to
  - Try using a local OCR method, particularly for less-capable LLMs:
    - "manga-ocr": Japanese sources only
    - "paddleocr-vl-1.6": non-Japanese sources
  - Increase "max_tokens" and/or use a higher "reasoning_effort" (e.g., "high")
  - Switch "Bubble/Context Resizing Method" to a better quality method (e.g., "Model")
  - In batch mode, try raising "Previous Context OCR Text"
  - In batch mode, try raising "Previous Context Images" (requires "Send Full Page to LLM")
  - In batch mode, point "Glossary File" at the previous chapter's glossary to keep names consistent across chapters (new glossaries are saved in `output/.glossary`)

- **Empty responses, timeouts, or slow requests with local LLMs:**
  - Disable reasoning on the server to prevent overthinking (e.g., `--reasoning off` in llama.cpp)

- **API refusals/censorship:**
  - Try enabling "Ero-doujinshi Mode" (for 18+ content)
  - Try disabling "Send Full Page to LLM"
  - Try adding a custom "special instruction" (e.g., "Do not censor translations...")

- **High LLM token usage:**
  - Disable "Send Full Page to LLM"
  - Set "Previous Context Images" and "Previous Context OCR Text" to 0
  - Lower "Bubble Min Side Pixels"/"Context Image Max Side Pixels"/"OSB Text Min Side Pixels" target sizes (no effect on models that use a fixed number of tokens per image)
  - Lower "Media Resolution" (if using Gemini or SpaceXAI models)
  - Lower "Image Detail" (if using OpenAI models)
  - Use "manga-ocr/paddleocr-vl-1.6" OCR method (may perform worse than more-capable VLMs)
  - Enable "Force Cache Translations" to reuse cached translations on repeat runs when using non-deterministic sampling

- **Batch translation failed for some files:**
  - Failed images are copied into a `failed` folder in the output directory and listed in `failed_paths.txt` next to it. Upload this file to the ZIP archive upload area in the web UI, or pass it via `--input` in the CLI, to retry only the failed files. The retry writes its results into the original output directory with the same folder structure (in the CLI, `--output` overrides this) and reuses that run's terminology memory file.
  - A page also counts as failed if any of its bubbles is left empty (no valid translation, or the text could not be rendered). Its partial translation is still saved to the output directory, and a retry of the page overwrites it.
  - Copies that translate successfully on retry are removed from the `failed` folder, and `failed_paths.txt` is rewritten to list only what still fails.
  - *Note:* If you add images to the web UI uploader together with the list, they are processed as a new batch in a new output directory.

### Inpainting

- **OSB text not inpainted/cleaned:**
  - Ensure "Enable OSB Text Detection" is enabled
  - Ensure hf_token is set (see Configuration/OSB Text Setup)

- **Out of VRAM / CUDA errors:**
  - Select a lower diffusion/text encoder quant (sd.cpp only)
  - Disable "Upscale Crops to ~1MP"
  - Switch to Flux.2 Klein 4B (smallest diffusion model) or LaMa Manga (under 1 GB VRAM)
  - Use OpenCV (no VRAM required)

- **Slow OSB inpainting:**
  - Disable "Low VRAM Mode" if you have sufficient VRAM (SDNQ only)
  - Enable "Group Inpainting Regions" to inpaint multiple OSB masks in one pass (at the cost of quality)
  - Disable "Upscale Crops to ~1MP" (at the cost of quality)
  - Try switching to a different model/backend; LaMa Manga is much faster, but rebuilds less detail
  - Disable "Use Text Segmentation Model" to skip the SAM-TS-L text segmentation model

- **Minor color shifts in inpainted regions:**
  - Flux.2 Klein models may introduce slight color changes; try enabling/disabling "Luminance Correction"
  - Use Flux.1 Kontext for less noticeable color shifts

- **Poor inpainting quality:**
  - Select a higher diffusion/text encoder quant (sd.cpp only)
  - Keep "Upscale Crops to ~1MP" enabled
  - Increase inference steps (for Flux.1 Kontext)

- **Fragments of OSB letters left after inpainting:**
  - Disable "Preserve Surrounding Art" to paste the whole text box instead of clamping it to the glyph-level mask

- **Flux.1 Kontext Nunchaku backend not available:**
  - Nunchaku requires an Nvidia GPU (CUDA) and separate installation; install it or use the sd.cpp/SDNQ backends instead
