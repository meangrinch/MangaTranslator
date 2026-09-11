import argparse
import os

# Suppress OpenBLAS NUM_THREADS warnings on machines with >24 logical cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import sys
from pathlib import Path
from threading import Thread


def custom_except_hook(gr, exc_type, exc_value, exc_traceback):
    from utils.logging import log_message

    if issubclass(exc_type, gr.Error):
        log_message(f"Gradio-handled Error: {exc_value}", always_print=True)
    else:
        import traceback

        log_message("--- Uncaught Exception ---", always_print=True)
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in "".join(tb_lines).splitlines():
            log_message(line, always_print=True)
        log_message("--------------------------", always_print=True)


def _get_pytorch_version_tuple(version_str):
    """Convert version string like '2.9.0' to tuple (2, 9, 0) for comparison."""
    parts = version_str.split("+")[0].split("-")[0].split(".")
    return tuple(int(part) for part in parts[:3])


def main():
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
    from utils.logging import init_file_logging, log_message

    init_file_logging()

    import gradio as gr
    import torch

    from core._version import __version__
    from core.device import get_best_device
    from ui import layout
    from utils.update_checker import check_and_display_changelog, check_for_update

    sys.excepthook = lambda exc_type, exc_value, exc_traceback: custom_except_hook(
        gr, exc_type, exc_value, exc_traceback
    )

    # PyTorch 2.9.0+ uses PYTORCH_ALLOC_CONF, older versions use PYTORCH_CUDA_ALLOC_CONF
    _pytorch_version = _get_pytorch_version_tuple(torch.__version__)
    if _pytorch_version >= (2, 9, 0):
        # Helps prevent fragmentation OOM errors on some GPUs
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    MODELS_DIR = Path(args.models)
    FONTS_BASE_DIR = Path(args.fonts)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FONTS_BASE_DIR, exist_ok=True)

    target_device = torch.device("cpu") if args.cpu else get_best_device()

    device_info_str = "CPU"
    if target_device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            device_info_str = f"CUDA ({gpu_name}, ID: 0)"
        except Exception:
            device_info_str = "CUDA (Unknown GPU Name)"
    elif target_device.type == "xpu":
        try:
            gpu_name = torch.xpu.get_device_name(0)
            device_info_str = f"XPU ({gpu_name}, ID: 0)"
        except Exception:
            device_info_str = "XPU (Unknown GPU Name)"
    elif target_device.type == "mps":
        device_info_str = "MPS (Apple Silicon)"
    log_message(f"Using device: {device_info_str.upper()}", always_print=True)
    log_message(f"PyTorch version: {torch.__version__}", always_print=True)
    log_message(f"MangaTranslator version: v{__version__}", always_print=True)
    check_and_display_changelog(__version__)

    def _update_notice():
        available, latest = check_for_update(
            __version__, repo="meangrinch/MangaTranslator", timeout=3.0
        )
        if available and latest:
            log_message(f"UPDATE AVAILABLE: v{latest.lstrip('v')}", always_print=True)

    Thread(target=_update_notice, daemon=True).start()

    app = layout.create_layout(
        models_dir=MODELS_DIR,
        fonts_base_dir=FONTS_BASE_DIR,
        target_device=target_device,
    )

    app.queue()
    app.launch(inbrowser=args.open_browser, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
