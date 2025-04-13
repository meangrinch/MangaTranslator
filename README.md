# MangaTranslator

Translate manga/comics speech bubbles using AI (YOLO for detection, LLMs for translation). Features a Gradio Web UI and CLI.

## Features

*   Automatic speech bubble detection &amp; segmentation.
*   Text removal &amp; cleaning from detected bubbles.
*   Text extraction &amp; translation via vision-capable LLMs.
*   Renders translated text onto images with selected fonts.
*   Web Interface (Gradio) &amp; Command-Line Interface (CLI).

## Requirements

*   Python >= 3.10
*   YOLO model with mask segmentation (trained for speech bubbles)
*   Vision-capable LLM (API or local)

## Installation

### Windows Portable

Download the standalone zip (NVIDIA GPU or CPU) from the [releases page](https://github.com/meangrinch/MangaTranslator/releases). 

*Includes recommended YOLO model and Komika font pack.*

### Manual Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/meangrinch/MangaTranslator.git
    cd MangaTranslator
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    # Create venv
    python -m venv venv
    
    # Activate (Windows CMD/PowerShell)
    .\venv\Scripts\activate
    
    # Activate (Linux/macOS/Git Bash)
    source venv/bin/activate
    ```

3.  **Install PyTorch:**
    ```python
    # Example (CUDA 12.4)
    pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

    # Example (CPU)
    pip install torch
    ```
    *Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for system-specific commands.*

4.  **Install Dependencies:**
    ```python
    pip install -r requirements.txt
    ```

## Post-Installation Setup

1.  **Download YOLO Model:**
    *   Download the [recommended model](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt) and place it in the `models` directory.

2.  **Prepare Fonts:**
    *   Place font folders (containing `.otf`/`.ttf` files) inside `fonts/`.
    *   Font variants need 'italic' or 'bold' in their filename to be used for emphasis. 
    *   Example structure:
        ```
        fonts/
        ├── CC Wild Words/
        │   ├── CC Wild Words Roman.otf
        │   ├── CC Wild Words Italic.otf
        │   ├── CC Wild Words Bold.otf
        │   └── CC Wild Words Bold Italic.otf
        └── Another Font/
            ├── AnotherFont-Regular.ttf
            └── AnotherFont-BoldItalic.ttf
        ```
    *Note: "CC Wild Words" is a common manga translation font.*

3.  **Setup LLM:**
    *   Supports external providers (Gemini, OpenAI, etc.,) and local models (Ollama, LMStudio, etc.,).
    *   **Web UI:** Configure in the "Config" tab (API keys saved locally to `config.json`).
    *   **CLI:** Pass API keys/endpoints as arguments.
    
    *Note: Environment variables (e.g., `GEMINI_API_KEY`) can also be used. See the "Config" tab for details.*

## Running

### **Web UI (Gradio):**
Use `start-webui.bat` or run `python app.py --open-browser`

*Note: First launch will take longer to open (~1-2 minutes).*

### **CLI:**
```python
# Example (Single - Gemini): 
python main.py --input <image_path> --yolo-model <model_path> --provider Gemini --gemini-api-key <key>

# Example (Batch - Ollama): 
python main.py --input <folder_path> --batch --yolo-model <model_path> --font-dir <custom_font_dir> --provider OpenAI-compatible --openai-compatible-url <url> --output <custom_output_folder>

# See all options: 
python main.py --help
```

## Basic Usage (Web UI)

1.  Launch the Web UI.
2.  Use the **"Translator"** (single image) or **"Batch"** (multiple) tab.
3.  Upload image(s).
4.  Select **Font**, **Source Language**, **Target Language**.
5.  Go to **"Config"** tab:
    *   Set **Translation -> LLM Provider**, **Model**, **API Key/Endpoint**.
    *   Set **Detection -> Reading Direction** (rtl/ltr).
    *   Click **"Save Config"** (Optional).
6.  Return to the previous tab and click **"Translate"** / **"Start Batch Translating"**.
7.  Output is saved to `./output/` by default.

*Note: A "cleaning only" mode is also available in the "Other" sub-tab.*

## Updating

Navigate to the `MangaTranslator` directory and run:
```bash
git pull
```

## Customization

Place custom YOLO models (`.pt`/`.onnx`) in `models/` (if using web ui). Must support segmentation and be trained for speech bubbles.

## License

GPL-3.0. See [LICENSE](https://github.com/meangrinch/MangaTranslator/blob/main/LICENSE).

## Author

grinnch