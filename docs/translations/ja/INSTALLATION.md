# インストール

## ポータブルパッケージ

[Releases](https://github.com/meangrinch/MangaTranslator/releases/tag/portable) からダウンロードします。

### システム要件

- **Windows：** PythonとGitが同梱されているため、追加の依存関係は不要です。
- **Linux/macOS：** システムにPython 3.10以上およびGitがインストールされている必要があります。

### アップデート

- **Windows：** ポータブルパッケージのルートから `update.bat` を実行
- **Linux/macOS：** ポータブルパッケージのルートから `./update.sh` を実行

> [!TIP]
> 新しいポータブルパッケージに移行する場合：
>
> - `fonts`、`models`、`output` ディレクトリは新しいポータブルパッケージにそのまま移動できます
> - `runtime` ディレクトリも、破損していなければそのまま移動を試すことができます

---

## 手動インストール

### 1. リポジトリのクローンと移動

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. PyTorchのインストール

システムに応じたPyTorchビルドをインストールします（[PyTorchインストールガイド](https://pytorch.org/get-started/locally/) を参照）：

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

### 4. Nunchakuのインストール（任意）

Nunchakuバックエンド経由でFLUX.1 Kontextを使用する場合に必要です。CUDA専用で、RTX 2000シリーズ以降のグラフィックカードが必要です。NunchakuのwheelファイルはGitHubリリースから直接インストールします：

```bash
# Windows (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Linux (Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> sd.cpp または SDNQ バックエンド経由で Flux を使用する場合、Nunchakuは不要です。

### 5. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### アップデート

リポジトリルートで実行：

```bash
git pull
pip install -r requirements.txt
```
