# 설치

## 포터블 패키지

[Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)에서 다운로드합니다.

### 시스템 요구 사항

- **Windows:** Python 및 Git이 내장되어 있어 별도의 의존성이 필요하지 않습니다.
- **Linux/macOS:** 시스템에 Python 3.10+ 및 Git이 설치되어 있어야 합니다.

### 업데이트

- **Windows:** 포터블 패키지 루트에서 `update.bat` 실행
- **Linux/macOS:** 포터블 패키지 루트에서 `./update.sh` 실행

> [!TIP]
> 새로운 포터블 패키지로 데이터를 이전해야 하는 경우:
>
> - `fonts`, `models`, `output` 디렉토리를 새 포터블 패키지로 안전하게 이동할 수 있습니다.
> - 손상되지 않은 경우 `runtime` 디렉토리도 이동을 시도할 수 있습니다.

---

## 수동 설치

### 1. 저장소 클론 및 이동

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. PyTorch 설치

시스템에 맞는 PyTorch 빌드를 설치합니다 ([PyTorch 설치 가이드](https://pytorch.org/get-started/locally/) 참조):

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

### 4. Nunchaku 설치 (선택 사항)

Nunchaku 백엔드를 통한 FLUX.1 Kontext에 필요합니다. CUDA 전용이며 RTX 2000 시리즈 이상의 그래픽 카드가 필요합니다. Nunchaku wheel 파일은 GitHub 릴리스에서 직접 설치합니다:

```bash
# Windows (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Linux (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> sd.cpp 또는 SDNQ 백엔드를 통해 Flux를 사용할 경우 Nunchaku가 필요하지 않습니다.

### 5. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 업데이트

저장소 루트에서 실행:

```bash
git pull
pip install -r requirements.txt
```
