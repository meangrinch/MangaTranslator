# Troubleshooting

### Portable Package Setup

- **Setup script fails to detect GPU:**
  - **NVIDIA:** Ensure NVIDIA drivers are installed and `nvidia-smi` is accessible from command line
  - **AMD (Windows/Linux):** Ensure ROCm is installed (AMD uses PyTorch's CUDA interface via ROCm)
  - **Intel ARC (Windows/Linux):** Ensure Intel GPU drivers are installed; uses XPU backend
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

- **PyTorch installation fails:**
  - Check your internet connection
  - Ensure you have enough disk space (~6 GB)
  - For ROCm: Ensure ROCm is properly installed on your system
  - Try running setup again; temporary network issues are common

- **Nunchaku installation fails:**
  - Nunchaku requires NVIDIA CUDA and Python 3.13
  - If installation fails, the app will use OpenCV inpainting instead
  - You can skip Nunchaku during setup and add it later

### Portable Package Updates

- **Update script fails:**
  - Check your internet connection
  - Ensure Git is installed and accessible (Linux/macOS)
  - Try running the update script again
  - If issues persist, re-download the portable package

### Rendering

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

### Detection/Cleaning

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

### Translation

- **Poor translations:**
  - Try "two-step" translation mode for less-capable LLMs
  - Try disabling "Send Full Page to LLM"
  - Try using "manga-ocr" OCR method, particularly for less-capable LLMs (Japanese sources only)
  - Increase "max_tokens" and/or use a higher "reasoning_effort" (e.g., "high")
  - Switch "Bubble/Context Resizing Method" to a better quality method (e.g., "Model")
- **API refusals/censorship:**
  - Try disabling "Send Full Page to LLM"
  - Try adding a custom "special instruction" (e.g., "Do not censor translations...")
- **High LLM token usage:**
  - Disable "Send Full Page to LLM"
  - Lower "Bubble Min Side Pixels"/"Context Image Max Side Pixels"/"OSB Min Side Pixels" target sizes
  - Lower "Media Resolution" (if using Gemini models)
  - Use "manga-ocr" OCR method (Japanese sources only; may perform worse than more-capable VLMs)

### Inpainting

- **OSB text not inpainted/cleaned:**
  - Ensure "Enable OSB Text Detection" is enabled
  - Ensure hf_token is set if using Kontext with Nunchaku backend (see Installation/Post-Install Setup)
- **Flux model too large/slow:**
  - Try Flux.2 Klein 4B (fastest/smallest model)
  - Enable "Low VRAM Mode" if you have limited VRAM (refer to VRAM chart in README)
  - Use OpenCV for maximum speed (lowest quality)
- **Out of VRAM / CUDA errors:**
  - Klein 9B (Full): requires ~12 GB VRAM; Klein 4B (Full): requires ~8 GB VRAM
  - Kontext SDNQ (Full): requires ~12 GB VRAM; Kontext Nunchaku: requires ~6 GB VRAM
  - Enable "Low VRAM Mode" for Klein or Kontext SDNQ to reduce to ~4 GB (slower)
  - Use OpenCV (no VRAM required)
- **Minor color shifts in inpainted regions:**
  - Klein models may introduce slight color/luminance changes
  - Use Flux Kontext (SDNQ or Nunchaku) for no color shifts
- **Flux Kontext Nunchaku backend not available:**
  - Nunchaku requires NVIDIA CUDA and separate installation
  - Use the SDNQ backend (cross-platform) or Flux Klein instead
- **Poor inpainting quality:**
  - Increase inference steps (Klein: 4-8, Kontext: 6-15)
  - Try Kontext over Klein for higher quality
