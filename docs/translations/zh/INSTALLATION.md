# 安装

## 便携版

从 [Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable) 下载。

### 系统要求

- **Windows：** 内置 Python 和 Git，无需额外依赖。
- **Linux/macOS：** 系统必须安装 Python 3.10+ 和 Git。

### 更新

- **Windows：** 在便携版根目录下运行 `update.bat`
- **Linux/macOS：** 在便携版根目录下运行 `./update.sh`

> [!TIP]
> 如果需要迁移到新的便携版：
>
> - 可以安全地将 `fonts`、`models` 和 `output` 目录移动到新的便携版中
> - 在没有损坏的前提下，也可以尝试将 `runtime` 目录复制过去

---

## 源码安装

### 1. 克隆并进入仓库

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

### 2. 创建并激活虚拟环境

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装 PyTorch

根据系统安装对应的 PyTorch 版本（参见：[PyTorch 安装指南](https://pytorch.org/get-started/locally/)）：

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

### 4. 安装 Nunchaku（可选）

通过 Nunchaku 后端使用 FLUX.1 Kontext 时需要。仅支持 CUDA，需要 RTX 2000 系列或更高版本的显卡。Nunchaku wheel 包直接从 GitHub releases 安装：

```bash
# Windows (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Linux (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> 通过 sd.cpp 或 SDNQ 后端使用 Flux 时不需要 Nunchaku。

### 5. 安装依赖

```bash
pip install -r requirements.txt
```

### 更新

在仓库根目录下运行：

```bash
git pull
pip install -r requirements.txt
```
