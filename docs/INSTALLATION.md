# Installation

## Portable Package

Download from [Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable).

### Requirements

- **Windows:** Python and Git are bundled; no additional dependencies required.
- **Linux/macOS:** Python 3.10+ and Git must be installed on your system

### Updating

- **Windows:** Run `update.bat` from the portable package root
- **Linux/macOS:** Run `./update.sh` from the portable package root

> [!TIP]
> In the event that you need to transfer to a fresh portable package:
>
> - You can safely move the `fonts`, `models`, and `output` directories to the new portable package
> - You can attempt to move the `runtime` directory over, assuming it isn't corrupted

---

## Manual Install

### 1. Clone and Enter the Repo

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install PyTorch

Install the PyTorch build for your system (see: [PyTorch Install](https://pytorch.org/get-started/locally/)):

```bash
# NVIDIA CUDA 13.0
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130

# AMD ROCm 7.1
pip install torch==2.11.0+rocm7.1 torchvision==0.26.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Intel XPU
pip install torch==2.11.0+xpu torchvision==0.26.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu

# Apple Silicon (MPS) / CPU
pip install torch==2.11.0 torchvision==0.26.0
```

### 4. Install Nunchaku (Optional)

Required for FLUX.1 Kontext via the Nunchaku backend. CUDA only, requires an RTX 2000-series card or newer. Nunchaku wheels are installed directly from GitHub releases:

```bash
# Windows (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Linux (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> Nunchaku is not required when using Flux via the sd.cpp or SDNQ backends.

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### Updating

From the repo root:

```bash
git pull
pip install -r requirements.txt
```
