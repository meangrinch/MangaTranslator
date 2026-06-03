# 常见问题与故障排除

### 便携版设置 (Portable Package Setup)

- **设置脚本未能检测到 GPU：**
  - **NVIDIA:** 确保已安装 NVIDIA GPU 驱动程序，并且可以在命令行访问 `nvidia-smi`
  - **AMD (Windows/Linux):** 确保已安装 AMD GPU 驱动程序
  - **Intel ARC (Windows/Linux):** 确保已安装 Intel GPU 驱动程序
  - **macOS Apple Silicon:** M1/M2/M3/M4 Mac 会自动检测并使用 MPS
  - **macOS Intel (AMD 独显):** 配备 AMD 独立显卡的 Intel Mac 可以使用 MPS
  - **macOS Intel (集显):** 使用集成显卡的 Intel Mac 只能运行在仅 CPU 模式下
  - 如果 GPU 检测失败，您随时可以选择 CPU 模式

- **“Python not found”（未找到 Python）错误 (Linux/macOS)：**
  - 安装 Python 3.10 或更高版本：
    - Ubuntu/Debian: `sudo apt install python3 python3-pip python3-venv`
    - Fedora: `sudo dnf install python3 python3-pip`
    - Arch: `sudo pacman -S python python-pip`
    - macOS: `brew install python@3.13` 或从 python.org 下载安装包

- **“Git not found”（未找到 Git）错误 (Linux/macOS)：**
  - 安装 Git：
    - Ubuntu/Debian: `sudo apt install git`
    - Fedora: `sudo dnf install git`
    - Arch: `sudo pacman -S git`
    - macOS: `xcode-select --install` 或 `brew install git`

- **Nunchaku 安装失败：**
  - Nunchaku 需要 NVIDIA CUDA 和 Python 3.10+
  - 如果安装失败，其他重绘方法仍然可用

### 渲染 (Rendering)

- **不支持的字符被移除：**
  - 您使用的字体不支持所选的目标语言；请使用支持目标语言的字体

- **错误的阅读顺序：**
  - 设置正确的“阅读方向 (Reading Direction)”（漫画通常为从右向左 `rtl`，美漫等为从左向右 `ltr`）
  - 对于能力较弱的 LLM，尝试使用“两步 (two-step)”模式或禁用“发送整页给 LLM (Send Full Page to LLM)”
  - 确保启用了“面板感知排序 (Use Panel-aware Sorting)”

- **文字太大或太小：**
  - 调整“最大字号 (Max Font Size)”和“最小字号 (Min Font Size)”的范围

- **为不同的气泡类型使用不同的字体（例如：心理活动 vs. 说话）：**
  - 用您想要的任何字体文件替换该字体的斜体（italic）/粗体（bold）/粗斜体（bold+italic）变体，只需确保文件名中包含这些关键字，以便程序能检测到变体
  - 要进一步微调结果，可以添加一条“特殊说明 (special instruction)”，指示 LLM 在内心独白等情况下使用斜体/粗体/粗斜体（这现在对应于您的自定义字体）

- **文字之间相互重叠：**
  - 您的字体文件可能已损坏；请尝试使用其他字体（例如便携版中附带的字体）

- **带变音符号的字符（如重音符号）未显示：**
  - 使用支持变音符号的字体（例如便携版中附带的 _Roboto_ 字体包，支持拉丁文、西里尔文和希腊文的变音符号）

- **文字过于模糊或有像素锯齿：**
  - 提高字体渲染的“超采样因子 (Supersampling Factor)”（例如 6-8）
  - 启用“初始 (initial)”图像超分辨率，并调整放大倍数（例如 2.0-4.0x）

- **空白气泡：**
  - 尝试降低“内边距像素 (Padding Pixels)”（例如 3-4）
  - 尝试降低“最小字号 (Min Font Size)”（例如 6-7）

- **气泡外替换文本过小或过于拥挤：**
  - 提高“窄长型放大乘数 (Narrow/Tall Expansion Multiplier)”（例如 2.0）和/或调整对应的阈值
  - 提高“微型放大乘数 (Tiny Expansion Multiplier)”（例如 2.0）和/或调整对应的阈值

### 检测与擦除 (Detection/Cleaning)

- **未擦除干净的文本残留（气泡边缘附近）：**
  - 降低“固定阈值 (Fixed Threshold Value)”（例如 170-190）和/或减少“收缩阈值 ROI (Shrink Threshold ROI)”（例如 0-2）

- **擦除时边缘轮廓被破坏/蚕食：**
  - 增加“收缩阈值 ROI (Shrink Threshold ROI)”（例如 6–8）
  - 增加“固定阈值 (Fixed Threshold Value)”（例如 210-220）

- **未检测到相连的对话框：**
  - 确保启用了“相连气泡检测 (Conjoined Bubble Detection)”
  - 降低“气泡检测置信度 (Bubble Detection Confidence)”（例如 0.20）

- **未检测到微小气泡/渲染文本空间不足：**
  - 启用“初始 (initial)”图像超分辨率并调整放大倍数（例如 2.0-4.0x），同时禁用“自动缩放 (Auto Scale)”

- **彩色/灰度气泡未能保留内部颜色：**
  - 启用“使用 Flux 重绘彩色气泡 (Use Flux to Inpaint Colored Bubbles)”

- **气泡分割效果较差：**
  - 尝试切换到 `sam3`（需要设置 `hf_token`）

### 翻译 (Translation)

- **翻译质量较差：**
  - 对于能力较弱的 LLM，尝试使用“两步 (two-step)”翻译模式
  - 尝试禁用“发送整页给 LLM (Send Full Page to LLM)”
  - 尝试使用本地 OCR 方法，特别是对于能力较弱 of LLM：
    - "manga-ocr": 仅限日语源文本
    - "paddleocr-vl": 非日语源文本
  - 增加 `max_tokens` 和/或使用更高的“推理努力 (reasoning_effort)”（例如 "high"）
  - 将“气泡/上下文缩放方法 (Bubble/Context Resizing Method)”切换到质量更好的方法（例如 "Model"）
  - 在批量模式下，尝试提高“前文 OCR 文本 (Previous Context OCR Text)”
  - 在批量模式下，尝试提高“前文图像 (Previous Context Images)”（需要开启“发送整页给 LLM”）

- **API 拒绝/审查拦截：**
  - 尝试禁用“发送整页给 LLM (Send Full Page to LLM)”
  - 尝试添加自定义的“特殊说明 (special instruction)”（例如，“不要审查翻译……”）

- **LLM Token 消耗量过高：**
  - 禁用“发送整页给 LLM (Send Full Page to LLM)”
  - 将“前文图像 (Previous Context Images)”和“前文 OCR 文本 (Previous Context OCR Text)”设置为 0
  - 降低“气泡最小边长 (Bubble Min Side Pixels)”/“上下文图像最大边长 (Context Image Max Side Pixels)”/“OSB 最小边长 (OSB Min Side Pixels)”的目标大小
  - 降低“媒体分辨率 (Media Resolution)”（如果使用 Gemini 或 xAI 模型）
  - 降低“图像细节 (Image Detail)”（如果使用 OpenAI 模型）
  - 使用 "manga-ocr/paddleocr-vl" OCR 方法（其效果可能不及更强的多模态模型）

### 重绘 (Inpainting)

- **对话框外 (OSB) 的文本未被重绘/擦除：**
  - 确保启用了“启用 OSB 文本检测 (Enable OSB Text Detection)”
  - 确保已设置 `hf_token`（参见 安装/安装后设置）

- **显存不足 (Out of VRAM) / CUDA 错误：**
  - 启用“低显存模式 (Low VRAM Mode)”（仅限 SDNQ）
  - 选择较低的 Flux/文本编码器量化版本（仅限 sd.cpp）
  - 禁用“将 Klein 裁剪块放大到约 100万像素 (Upscale Klein Crops to ~1MP)”
  - 切换至 Flux.2 Klein 4B（最小的模型）
  - 使用 OpenCV（不需要显存）

- **Flux OSB 重绘速度过慢：**
  - 启用“合并 Flux 区域 (Group Flux Regions)”以在单次 Flux 运行中重绘多个 OSB 掩码（代价是会降低质量）
  - 禁用“将 Klein 裁剪块放大到约 100万像素 (Upscale Klein Crops to ~1MP)”（代价是会降低质量）
  - 尝试切换到其他模型/后端

- **重绘区域出现轻微色差：**
  - Flux.2 Klein 模型可能会引入轻微的颜色变化；尝试启用/禁用“亮度校正 (Luminance Correction)”
  - 使用 Flux.1 Kontext 以获得不明显的色差效果

- **重绘质量较差：**
  - 选择较高的 Flux/文本编码器量化版本（仅限 sd.cpp）
  - 保持启用“将 Klein 裁剪块放大到约 100万像素 (Upscale Klein Crops to ~1MP)”
  - 增加推理步数（对于 Flux.1 Kontext）

- **Flux.1 Kontext Nunchaku 后端不可用：**
  - Nunchaku 需要 Nvidia GPU (CUDA) 且需单独安装；请进行安装，或者改为使用 sd.cpp/SDNQ 后端
