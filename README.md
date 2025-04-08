# MangaTranslator

Automatically translate speech bubbles in manga/comics using AI. This tool utilizes YOLO for speech bubble detection, Large Language Models (LLMs) for translation, and Gradio for the web interface.

## Features

*   Automatic speech bubble detection and segmentation.
*   Text removal and intelligent filling (i.e. cleaning) from detected bubbles (white or black).
*   Text extraction and translation via a vision-capable LLM.
*   Rendering translated text back onto the image with selected fonts.
*   Easy-to-use Web Interface (Gradio)
*   Command-Line Interface (CLI)    

## Requirements

*   Python >= 3.10
*   A YOLO model with mask segmentation (must be trained for speech bubble detection)
*   An LLM with Vision capabilities (API or local)
*   (Optional) NVIDIA GPU with CUDA support

# Installation

## Windows Portable

Download the standalone zip for NVIDIA GPUs (or CPU only) on the [releases page](https://github.com/meangrinch/MangaTranslator/releases), which includes the recommended YOLO model and an example font pack.

## Manual Install

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/meangrinch/MangaTranslator.git
    cd MangaTranslator
    ```

2.  **Create & Activate Virtual Environment (Recommended):**
    ```bash
    # Create venv
    python -m venv venv

    # Activate (Windows CMD/PowerShell)
    .\venv\Scripts\activate

    # Activate (Linux/macOS/Git Bash)
    source venv/bin/activate
    ```

3.  **Install PyTorch:**
    *   Example for NVIDIA GPU with CUDA 12.4:
        ```bash
        pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
        ```
    *   Example for CPU:
        ```bash
        pip install torch torchvision
        ```
    *   *(Note: Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for commands specific to your system configuration if needed.)*

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download YOLO Detection Model:**
    *   Download the [recommended](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model.pt) YOLO model
    *   Place the downloaded `.pt` (or `.onnx`) file inside the `models` directory.

6.  **Prepare Fonts:**
    *   Place font folder(s) inside `fonts/`. Each sub-folder represents a font pack and should contain the `.ttf` or `.otf` files for that font. Example structure:
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
    *   Font variants (i.e. bold, etc.,) are selectively used by the LLM and *must* contain certain words ('italic', 'bold', or both) to be used.

*(Note: "CC Wild Words" is the most common font used for manga translations.)*

7.  **Setup Vision-capable LLM:**
    
You can either use an external LLM provider (e.g., Gemini, OpenAI, Anthropic, OpenRouter) via an API key or a local model via an OpenAI-compatible endpoint (e.g., Ollama, LMStudio).

*   **WebUI:**
    *   Go to the "Config" tab -> "Translation" section, select your provider, and enter your API key/OpenAI-compatible endpoint (url). Keys are saved locally in `config.json`.
*   **Command-Line:**
    *   For CLI usage, keys can be passed as arguments (see the CLI section below for details).

*Note: You can also set environment variables (like `GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.,). The UI's "Config" tab provides info on which variable corresponds to which provider.*

## Running the Application

*   **Web UI (Recommended):**
    *   **Windows:** Double-click `start-webui.bat` (assuming `venv` is present).
    *   **Linux/macOS:** Run `python app.py --open-browser` in your terminal.

*   **Command-Line Interface (CLI):**
    *   Use `python main.py` for single image or batch processing.
    *   **Example (Single Image - Gemini):**
        ```bash
        python main.py --input path/to/your/image.png --output path/to/save/result.png --yolo-model models/yolov8m_seg-speech-bubble.pt --provider Gemini --gemini-api-key YOUR_API_KEY --font-dir "fonts/CC Wild Words"
        ```
    *   **Example (Batch - OpenAI):**
        ```bash
        python main.py --input path/to/image_folder --output path/to/output_folder --batch --yolo-model models/yolov8m_seg-speech-bubble.pt --provider OpenAI --openai-api-key YOUR_API_KEY
        ```
    *   See all available options and their defaults:
        ```bash
        python main.py --help
        ```

## Basic Usage (Web UI)

1.  Launch the Web UI as described above.
2.  Go to the **"Translator"** tab for single images or **"Batch"** for multiple files/folders.
3.  Upload your manga/comic page image(s).
4.  Select the desired **Font**, **Source Language**, and **Target Language** in the dropdowns.
5.  Go to the **"Config"** tab to configure translation and processing settings:
    *   Under **"Translation"**: Select your **Translation Provider**, the specific **Model**, and enter your **API Key** (if not using environment variables). Adjust Temperature, Top-P, Top-K as needed.
    *   Under **"Detection"**: Set **Reading Direction** (rtl for Manga, ltr for Comics).
    *   **Important:** Click **"Save Config"** at the top of the Config tab to apply your changes.
6.  Return to the **"Translator"** or **"Batch"** tab and click the **"Translate"** / **"Start Batch Translating"** button.
7.  Results will appear on the right (single image) or in the gallery (batch). Status messages provide details about the process.
8.  Output images are saved in the `./output/` directory by default (or the specified output directory for CLI batch).

*Note: There is a "cleaning only" mode available that skips the translation and text rendering steps, outputting just the cleaned speech bubbles.*

## Customization

*   **Using Custom YOLO Models:** Place your custom YOLO model file (`.pt`) in the `models/` directory. The model must support segmentation and be trained to detect speech bubbles.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/meangrinch/MangaTranslator/blob/main/LICENSE) file for details.

## Author

grinnch