import argparse
import os
import sys
from pathlib import Path
from threading import Thread

import gradio as gr
import torch

import core
from ui import layout
from utils.update_checker import check_for_update


def custom_except_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, gr.Error):
        print(f"Gradio-handled Error: {exc_value}")
    else:
        import traceback

        print("--- Uncaught Exception ---")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("--------------------------")


sys.excepthook = custom_except_hook

# Helps prevent fragmentation OOM errors on some GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MangaTranslator")
    parser.add_argument(
        "--models",
        type=str,
        default="./models",
        help="Directory containing YOLO model files",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        default="./fonts",
        help="Base directory containing font pack subdirectories",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open in the default web browser",
    )
    parser.add_argument(
        "--port", type=int, default=7676, help="Port number for the web UI"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage even if CUDA is available"
    )
    args = parser.parse_args()

    MODELS_DIR = Path(args.models)
    FONTS_BASE_DIR = Path(args.fonts)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FONTS_BASE_DIR, exist_ok=True)

    target_device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    device_info_str = "CPU"
    if target_device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            device_info_str = f"CUDA ({gpu_name}, ID: 0)"
        except Exception:
            device_info_str = "CUDA (Unknown GPU Name)"
    print(f"Using device: {device_info_str.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MangaTranslator version: {core.__version__}")

    def _update_notice():
        available, latest = check_for_update(
            core.__version__, repo="meangrinch/MangaTranslator", timeout=3.0
        )
        if available and latest:
            print(f"UPDATE AVAILABLE: {latest}")

    Thread(target=_update_notice, daemon=True).start()

    app = layout.create_layout(
        models_dir=MODELS_DIR,
        fonts_base_dir=FONTS_BASE_DIR,
        target_device=target_device,
    )

    app.queue()
    app.launch(inbrowser=args.open_browser, server_port=args.port, show_error=True)
