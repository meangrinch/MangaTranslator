import contextlib
import logging
import os
import shutil
import threading
from enum import Enum
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from spandrel import ModelLoader
from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    Sam2Model,
    Sam2Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)
from ultralytics import YOLO

from core.device import empty_cache, get_best_device, get_best_dtype, get_device_info
from core.ml.rtdetr_adapter import RTDetrYOLOAdapter
from core.ml.sdcpp_server import SDCppServerManager
from utils.cancellation import raise_if_cancelled
from utils.download import download_file
from utils.exceptions import ModelError
from utils.logging import log_message
from utils.model_metadata import FLUX_SDCPP_QUANT_FILES


def _add_cancel_check_to_denoise_loops(blocks) -> None:
    """Raise CancellationError after each denoising step of a modular pipeline.

    Modular pipelines have no step callback or interrupt flag; a raising block is the
    documented way to stop them, and the pipeline stays usable afterwards.
    """
    from diffusers.modular_pipelines import (
        LoopSequentialPipelineBlocks,
        ModularPipelineBlocks,
    )

    class CancelCheck(ModularPipelineBlocks):
        @property
        def inputs(self):
            return []

        def __call__(self, components, block_state, i, t):
            raise_if_cancelled()
            return components, block_state

    def denoise_loops(block):
        if (
            isinstance(block, LoopSequentialPipelineBlocks)
            and "denoiser" in block.sub_blocks
        ):
            yield block
        for sub_block in getattr(block, "sub_blocks", {}).values():
            yield from denoise_loops(sub_block)

    for loop in denoise_loops(blocks):
        loop.sub_blocks["cancel_check"] = CancelCheck()


class ModelType(Enum):
    """Enumeration of available model types."""

    UPSCALE = "upscale"
    UPSCALE_LITE = "upscale_lite"
    YOLO_SPEECH_BUBBLE = "yolo_speech_bubble"
    YOLO_SPEECH_BUBBLE_2 = "yolo_speech_bubble_2"
    RTDETR_CONJOINED_BUBBLE = "rtdetr_conjoined_bubble"
    YOLO_OSBTEXT = "yolo_osbtext"
    YOLO_PANEL = "yolo_panel"
    SAM2 = "sam2"
    SAM3 = "sam3"
    MANGA_OCR = "manga_ocr"
    PADDLE_OCR_VL = "paddle_ocr_vl"
    FLUX_TRANSFORMER = "flux_transformer"
    FLUX_TEXT_ENCODER = "flux_text_encoder"
    FLUX_PIPELINE = "flux_pipeline"
    FLUX_KONTEXT_SDNQ_PIPELINE = "flux_kontext_sdnq_pipeline"
    FLUX_KLEIN_9B_PIPELINE = "flux_klein_9b_pipeline"
    FLUX_KLEIN_4B_PIPELINE = "flux_klein_4b_pipeline"
    FLUX3_KLEIN_SDNQ_PIPELINE = "flux3_klein_sdnq_pipeline"
    QWEN_IMAGE_21_SDNQ_PIPELINE = "qwen_image_21_sdnq_pipeline"
    LAMA_MANGA = "lama_manga"
    TEXT_SAM = "text_sam"
    SDCPP_SERVER = "sdcpp_server"
    FLUX2_SDCPP_VAE = "flux2_sdcpp_vae"
    QWEN_IMAGE_21_SDCPP_VAE = "qwen_image_21_sdcpp_vae"
    FLUX_KONTEXT_SDCPP_CLIP_L = "flux_kontext_sdcpp_clip_l"
    FLUX_KONTEXT_SDCPP_VAE = "flux_kontext_sdcpp_vae"


# Old flat folder -> its place in the task-grouped layout (flux/sdcpp moves before flux)
_LEGACY_MODEL_DIRS = (
    ("flux/sdcpp", "inpainting/sdcpp"),
    ("flux", "inpainting/diffusers"),
    ("lama", "inpainting/lama"),
    ("yolo", "detection/yolo"),
    ("rtdetr", "detection/rtdetr"),
    ("sam", "detection/sam"),
    ("manga-ocr-base", "ocr/manga-ocr-base"),
    ("paddleocr-vl-1.6", "ocr/paddleocr-vl-1.6"),
    ("sdcpp", "runtime/sdcpp"),
)


def migrate_model_dirs(models_dir: Path | str = "./models") -> None:
    """Move downloads from the old flat models/ layout into the task-grouped one.

    A folder is left where it is when its new location already exists.
    """
    models_dir = Path(models_dir)
    for old, new in _LEGACY_MODEL_DIRS:
        src, dst = models_dir / old, models_dir / new
        if not src.is_dir() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.rename(dst)
        except OSError as e:
            log_message(
                f"Could not move {src} to {dst} ({e}). Move it by hand to avoid "
                "downloading its models again.",
                always_print=True,
            )
            continue
        log_message(f"Moved {src} to {dst}", always_print=True)


class ModelManager:
    """Singleton model manager for MangaTranslator."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model manager (only once due to singleton pattern)."""
        with self._lock:
            if self._initialized:
                return

            self.device = get_best_device()
            self.dtype = get_best_dtype(self.device)

            self.models = {}
            self._apply_models_dir(Path("./models"))
            self.model_urls = self._init_model_urls()
            self.model_hf_repos = self._init_hf_repos()

            self.use_hf_cache = False
            self.hf_token = None
            self.flux_hf_token = None
            self.flux_residual_diff_threshold = 0.15
            self.sdcpp_server_manager = SDCppServerManager(
                self.model_paths[ModelType.SDCPP_SERVER]
            )

            # Serializes Flux pipeline inference across threads (CPU offload is not thread-safe)
            self.flux_inference_lock = threading.Lock()
            # SAM-TS-L sits in RAM after park_text_sam until load_text_sam returns it
            self._text_sam_parked = False

            self._initialized = True
            log_message(
                f"Model Manager initialized on device: {self.device}", always_print=True
            )

    def _apply_models_dir(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir).resolve()
        self.diffusers_cache_dir = self.models_dir / "inpainting" / "diffusers"
        self.sam_cache_dir = self.models_dir / "detection" / "sam"
        self.model_paths = self._init_model_paths()

    def _flux_sdcpp_base_dir(self) -> Path:
        """Base directory for sd.cpp model weights."""
        return self.models_dir / "inpainting" / "sdcpp"

    def _init_model_paths(self):
        """Initialize model file paths."""
        model_dir = self.models_dir
        detection_dir = model_dir / "detection"
        flux_sdcpp_dir = self._flux_sdcpp_base_dir()
        flux_kontext_sdcpp_dir = flux_sdcpp_dir / "kontext"
        return {
            ModelType.UPSCALE: (
                model_dir / "upscale" / "2x-AnimeSharpV4_RCAN.safetensors"
            ),
            ModelType.UPSCALE_LITE: (
                model_dir / "upscale" / "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
            ),
            ModelType.YOLO_SPEECH_BUBBLE: (
                detection_dir / "yolo" / "yolov8m_seg-speech-bubble.pt"
            ),
            ModelType.YOLO_SPEECH_BUBBLE_2: (
                detection_dir / "yolo" / "manga109-segmentation-bubble.pt"
            ),
            ModelType.RTDETR_CONJOINED_BUBBLE: (
                detection_dir / "rtdetr" / "comic-text-and-bubble-detector"
            ),
            ModelType.YOLO_OSBTEXT: (detection_dir / "yolo" / "animetext_yolov12x.pt"),
            ModelType.YOLO_PANEL: (
                detection_dir / "yolo" / "manga109_v2023.12.07_l_yolov11.pt"
            ),
            ModelType.LAMA_MANGA: (
                model_dir / "inpainting" / "lama" / "lama-manga.safetensors"
            ),
            ModelType.TEXT_SAM: (
                detection_dir / "text_sam" / "koharu-text-sam-ts-l.safetensors"
            ),
            ModelType.MANGA_OCR: (model_dir / "ocr" / "manga-ocr-base"),
            ModelType.PADDLE_OCR_VL: (model_dir / "ocr" / "paddleocr-vl-1.6"),
            ModelType.SDCPP_SERVER: (
                model_dir / "runtime" / "sdcpp" / SDCppServerManager.RELEASE_TAG
            ),
            ModelType.FLUX2_SDCPP_VAE: (flux_sdcpp_dir / "flux2-vae.safetensors"),
            ModelType.QWEN_IMAGE_21_SDCPP_VAE: (
                flux_sdcpp_dir / "qwen_image_2.1_vae_bf16.safetensors"
            ),
            ModelType.FLUX_KONTEXT_SDCPP_CLIP_L: (
                flux_kontext_sdcpp_dir / "clip_l.safetensors"
            ),
            ModelType.FLUX_KONTEXT_SDCPP_VAE: (
                flux_kontext_sdcpp_dir / "ae.safetensors"
            ),
        }

    def _init_model_urls(self):
        """Initialize model download URLs."""
        return {
            ModelType.UPSCALE: (
                "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
                "2x-AnimeSharpV4_RCAN.safetensors"
            ),
            ModelType.UPSCALE_LITE: (
                "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
                "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
            ),
            ModelType.FLUX2_SDCPP_VAE: (
                "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/"
                "split_files/vae/flux2-vae.safetensors"
            ),
            ModelType.QWEN_IMAGE_21_SDCPP_VAE: (
                "https://huggingface.co/unsloth/Qwen-Image-2.1-FP8/resolve/main/"
                "vae/qwen_image_2.1_vae_bf16.safetensors"
            ),
            ModelType.FLUX_KONTEXT_SDCPP_CLIP_L: (
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/"
                "clip_l.safetensors"
            ),
            ModelType.FLUX_KONTEXT_SDCPP_VAE: (
                "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/"
                "split_files/vae/ae.safetensors"
            ),
        }

    def _init_hf_repos(self):
        """Initialize Hugging Face repository information."""
        repos = {
            ModelType.UPSCALE: {
                "repo_id": "Kim2091/2x-AnimeSharpV4",
                "filename": "2x-AnimeSharpV4_RCAN.safetensors",
            },
            ModelType.UPSCALE_LITE: {
                "repo_id": "Kim2091/2x-AnimeSharpV4",
                "filename": "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            },
            ModelType.YOLO_SPEECH_BUBBLE: {
                "repo_id": "kitsumed/yolov8m_seg-speech-bubble",
                "filename": "model.pt",
            },
            ModelType.YOLO_SPEECH_BUBBLE_2: {
                "repo_id": "huyvux3005/manga109-segmentation-bubble",
                "filename": "best.pt",
            },
            ModelType.RTDETR_CONJOINED_BUBBLE: {
                "repo_id": "ogkalu/comic-text-and-bubble-detector",
            },
            ModelType.YOLO_OSBTEXT: {
                "repo_id": "deepghs/AnimeText_yolo",
                "filename": "yolo12x_animetext/model.pt",
            },
            ModelType.YOLO_PANEL: {
                "repo_id": "deepghs/manga109_yolo",
                "filename": "v2023.12.07_l_yv11/model.pt",
            },
            ModelType.SAM2: {
                "repo_id": "facebook/sam2.1-hiera-large",
            },
            ModelType.SAM3: {
                "repo_id": "facebook/sam3",
                "requires_token": True,
            },
            ModelType.FLUX_PIPELINE: {
                "repo_id": "black-forest-labs/FLUX.1-Kontext-dev",
                "filename": None,  # Pipeline loaded via from_pretrained
            },
        }

        repos[ModelType.FLUX_TRANSFORMER] = {
            "repo_id": "nunchaku-tech/nunchaku-flux.1-kontext-dev",
            "filename": None,  # Will be constructed dynamically in load_flux_models()
        }
        repos[ModelType.FLUX_TEXT_ENCODER] = {
            "repo_id": "nunchaku-tech/nunchaku-t5",
            "filename": "awq-int4-flux.1-t5xxl.safetensors",
        }
        repos[ModelType.MANGA_OCR] = {
            "repo_id": "kha-white/manga-ocr-base",
            "revision": "refs/pr/4",
        }
        repos[ModelType.PADDLE_OCR_VL] = {
            "repo_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        }
        # Flux.1 Kontext SDNQ (cross-platform, no token required)
        repos[ModelType.FLUX_KONTEXT_SDNQ_PIPELINE] = {
            "repo_id": "Disty0/FLUX.1-Kontext-dev-SDNQ-uint4-svd-r32",
        }
        # Flux.2 Klein models (SDNQ quantized, public)
        repos[ModelType.FLUX_KLEIN_9B_PIPELINE] = {
            "repo_id": "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        }
        repos[ModelType.FLUX_KLEIN_4B_PIPELINE] = {
            "repo_id": "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        }
        repos[ModelType.FLUX3_KLEIN_SDNQ_PIPELINE] = {
            "repo_id": "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_4bit",
        }
        repos[ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE] = {
            "repo_id": "SamuelTallet/Qwen-Image-2.1-Turbo-SDNQ-4bit-dynamic-hadamard256",
        }
        repos[ModelType.LAMA_MANGA] = {
            "repo_id": "mayocream/lama-manga",
            "filename": "lama-manga.safetensors",
        }
        repos[ModelType.TEXT_SAM] = {
            "repo_id": "mayocream/koharu-text-sam-ts-l",
            "filename": "model.safetensors",
        }
        # The FLUX.2 VAE, shared by Flux.2 Klein, Flux.3 Image Klein, and Fibo 2.0
        repos[ModelType.FLUX2_SDCPP_VAE] = {
            "repo_id": "Comfy-Org/flux2-dev",
            "filename": "split_files/vae/flux2-vae.safetensors",
        }
        repos[ModelType.QWEN_IMAGE_21_SDCPP_VAE] = {
            "repo_id": "unsloth/Qwen-Image-2.1-FP8",
            "filename": "vae/qwen_image_2.1_vae_bf16.safetensors",
        }
        repos[ModelType.FLUX_KONTEXT_SDCPP_CLIP_L] = {
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "clip_l.safetensors",
        }
        repos[ModelType.FLUX_KONTEXT_SDCPP_VAE] = {
            "repo_id": "Comfy-Org/Lumina_Image_2.0_Repackaged",
            "filename": "split_files/vae/ae.safetensors",
        }

        return repos

    def _enable_hf_xet_high_performance(self) -> None:
        if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    def _hf_cache_dir(self, local_dir: Path | str) -> str | None:
        """cache_dir for from_pretrained; None selects the default HF cache."""
        return None if self.use_hf_cache else str(local_dir)

    def _remove_empty_child_dirs(self, root: Path, verbose: bool = False) -> None:
        if not root.exists():
            return
        for path in sorted(
            (child for child in root.rglob("*") if child.is_dir()),
            key=lambda child: len(child.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
                log_message(
                    f"Removed empty download directory {path}.", verbose=verbose
                )
            except OSError:
                pass

    def _flux_sdcpp_quant_asset(
        self, model_key: str, asset_key: str, quant: str
    ) -> dict:
        quant_spec = FLUX_SDCPP_QUANT_FILES[model_key][asset_key]
        selected_quant = quant or quant_spec["default"]
        filenames = quant_spec["filenames"]
        if selected_quant not in filenames:
            options = ", ".join(filenames)
            raise ModelError(
                f"Invalid sd.cpp quant '{selected_quant}' for {model_key} "
                f"{asset_key}. Choose one of: {options}."
            )

        base_dir = self._flux_sdcpp_base_dir()
        if quant_spec.get("subdir"):
            base_dir = base_dir / quant_spec["subdir"]
        filename = filenames[selected_quant]
        asset = {
            "repo_id": quant_spec.get("quant_repos", {}).get(
                selected_quant, quant_spec["repo_id"]
            ),
            "filename": filename,
            "path": base_dir / Path(filename).name,
            "quant": selected_quant,
        }
        vision = quant_spec.get("vision")
        if vision is not None and filename.endswith(".gguf"):
            asset["vision"] = {
                "repo_id": vision["repo_id"],
                "filename": vision["filename"],
                "path": base_dir / vision["local_name"],
            }
        return asset

    def _flux_sdcpp_asset_spec(
        self,
        model_key: str,
        diffusion_quant: str = "",
        text_encoder_quant: str = "",
    ) -> dict:
        qwen_llm_models = {
            "diffusion_model": diffusion_quant,
            "llm": text_encoder_quant,
        }
        specs = {
            "flux_klein_4b": {
                "label": "Flux.2 Klein 4B",
                "quant_models": qwen_llm_models,
                "fixed_model_types": {"vae": ModelType.FLUX2_SDCPP_VAE},
                "server_args": ("--llm", "llm"),
            },
            "flux_klein_9b": {
                "label": "Flux.2 Klein 9B",
                "quant_models": qwen_llm_models,
                "fixed_model_types": {"vae": ModelType.FLUX2_SDCPP_VAE},
                "server_args": ("--llm", "llm"),
            },
            "flux_kontext": {
                "label": "Flux.1 Kontext",
                "quant_models": {
                    "diffusion_model": diffusion_quant,
                    "t5xxl": text_encoder_quant,
                },
                "fixed_model_types": {
                    "clip_l": ModelType.FLUX_KONTEXT_SDCPP_CLIP_L,
                    "vae": ModelType.FLUX_KONTEXT_SDCPP_VAE,
                },
                "server_args": ("--clip_l", "clip_l", "--t5xxl", "t5xxl"),
            },
            "flux3_klein": {
                "label": "Flux.3 Image Klein",
                "quant_models": qwen_llm_models,
                "fixed_model_types": {"vae": ModelType.FLUX2_SDCPP_VAE},
                "server_args": ("--llm", "llm"),
            },
            "fibo2_edit_turbo": {
                "label": "Fibo 2.0 Edit Turbo",
                "quant_models": qwen_llm_models,
                "fixed_model_types": {"vae": ModelType.FLUX2_SDCPP_VAE},
                "server_args": ("--llm", "llm"),
            },
            "qwen_image_2_1_turbo": {
                "label": "Qwen-Image 2.1 Turbo",
                "quant_models": qwen_llm_models,
                "fixed_model_types": {"vae": ModelType.QWEN_IMAGE_21_SDCPP_VAE},
                "server_args": ("--llm", "llm"),
                "extra_args": ("--sigmas", "qwen-image-2.1-turbo"),
            },
        }
        if model_key not in specs:
            raise ModelError(f"Unknown sd.cpp Flux model key: {model_key}")
        spec = specs[model_key]
        spec["quant_assets"] = {
            key: self._flux_sdcpp_quant_asset(model_key, key, quant)
            for key, quant in spec["quant_models"].items()
        }
        spec["extra_assets"] = {}
        vision = spec["quant_assets"].get("llm", {}).get("vision")
        if vision is not None:
            spec["extra_assets"]["llm_vision"] = vision
            spec["server_args"] = (*spec["server_args"], "--llm_vision", "llm_vision")
        return spec

    def ensure_flux_sdcpp_assets(
        self,
        model_key: str = "flux_klein_4b",
        diffusion_quant: str = "",
        text_encoder_quant: str = "",
        verbose: bool = False,
    ) -> dict:
        """Ensure sd.cpp server and Flux GGUF assets are available."""
        with self._lock:
            spec = self._flux_sdcpp_asset_spec(
                model_key,
                diffusion_quant=diffusion_quant,
                text_encoder_quant=text_encoder_quant,
            )
            quant_label = "/".join(
                asset["quant"] for asset in spec["quant_assets"].values()
            )
            server_quant_key = "-".join(
                asset["quant"] for asset in spec["quant_assets"].values()
            )
            log_message(
                f"Preparing {spec['label']} sd.cpp assets ({quant_label})...",
                verbose=verbose,
            )
            assets = {
                "executable": self.sdcpp_server_manager.ensure_server_executable(
                    verbose=verbose
                ),
                "model_key": model_key,
                "label": spec["label"],
                "server_args": spec["server_args"],
                "extra_args": spec.get("extra_args", ()),
                "server_model_key": f"{model_key}-{server_quant_key}",
            }
            for key, asset in (
                *spec["quant_assets"].items(),
                *spec["extra_assets"].items(),
            ):
                assets[key] = self._ensure_hf_file(
                    asset["repo_id"],
                    asset["filename"],
                    asset["path"],
                    token=self.flux_hf_token,
                    verbose=verbose,
                )

            for key, model_type in spec["fixed_model_types"].items():
                hf_info = self.model_hf_repos[model_type]
                assets[key] = self._ensure_hf_file(
                    hf_info["repo_id"],
                    hf_info["filename"],
                    self.model_paths[model_type],
                    token=self.flux_hf_token,
                    verbose=verbose,
                )

            return assets

    def ensure_flux_sdcpp_server(
        self,
        model_key: str = "flux_klein_4b",
        cache_mode: str = "none",
        diffusion_quant: str = "",
        text_encoder_quant: str = "",
        num_inference_steps: int = 4,
        verbose: bool = False,
    ) -> dict:
        """Start or reuse a persistent sd.cpp server for a Flux model."""
        assets = self.ensure_flux_sdcpp_assets(
            model_key,
            diffusion_quant=diffusion_quant,
            text_encoder_quant=text_encoder_quant,
            verbose=verbose,
        )
        return self.sdcpp_server_manager.ensure_flux_server(
            model_key,
            assets,
            cache_mode=cache_mode,
            num_inference_steps=num_inference_steps,
            verbose=verbose,
        )

    def shutdown_sdcpp_server(self, model_key: str, verbose: bool = False) -> None:
        with self.flux_inference_lock:
            self.sdcpp_server_manager.shutdown_server(model_key, verbose=verbose)

    def shutdown_sdcpp_servers(self, verbose: bool = False) -> None:
        with self.flux_inference_lock:
            self.sdcpp_server_manager.shutdown_servers(verbose=verbose)

    def _ensure_hf_file(
        self,
        repo_id: str,
        filename: str,
        target: Path,
        token: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download file from Hugging Face if it doesn't exist.

        Args:
            repo_id: Hugging Face repository ID
            filename: Name of file to download
            target: Path where file should be saved
            token: Optional Hugging Face token
            verbose: Whether to print verbose logging

        Returns:
            Path to the file (inside the HF cache when use_hf_cache is set)
        """
        if self.use_hf_cache:
            self._enable_hf_xet_high_performance()
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token or self.hf_token or os.environ.get("HF_TOKEN"),
                )
            )
        if target.exists():
            self._remove_empty_child_dirs(target.parent, verbose=verbose)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        log_message(
            f"Downloading {target.name} from Hugging Face ({repo_id})...",
            verbose=verbose,
        )
        self._enable_hf_xet_high_performance()
        effective_token = token or self.hf_token or os.environ.get("HF_TOKEN")
        if effective_token:
            os.environ["HF_TOKEN"] = effective_token
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target.parent),
            token=effective_token,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != target:
            try:
                downloaded_path.replace(target)
            except Exception:
                shutil.copyfile(downloaded_path, target)
                with contextlib.suppress(Exception):
                    downloaded_path.unlink()

            self._remove_empty_child_dirs(target.parent, verbose=verbose)
        log_message(f"Downloaded {target.name} successfully.", verbose=verbose)
        return target

    def _ensure_hf_repo(
        self,
        repo_id: str,
        target_dir: Path,
        token: str | None = None,
        revision: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download entire repository from Hugging Face if it doesn't exist.

        Args:
            repo_id: Hugging Face repository ID
            target_dir: Directory where repository should be saved
            token: Optional Hugging Face token
            revision: Optional git revision (branch, tag, commit hash)
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded repository directory
        """
        if self.use_hf_cache:
            self._enable_hf_xet_high_performance()
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    token=token or self.hf_token or os.environ.get("HF_TOKEN"),
                    revision=revision,
                )
            )
        # Check for safetensors for transformers v5+
        safetensors_file = target_dir / "model.safetensors"
        bin_file = target_dir / "pytorch_model.bin"

        is_downloaded = False
        if target_dir.exists() and (
            safetensors_file.exists() or (bin_file.exists() and revision is None)
        ):
            is_downloaded = True

        if is_downloaded:
            return target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        log_message(
            (
                f"Downloading repository {repo_id} (revision: {revision})..."
                if revision
                else f"Downloading repository {repo_id}..."
            ),
            verbose=verbose,
        )
        self._enable_hf_xet_high_performance()
        effective_token = token or self.hf_token or os.environ.get("HF_TOKEN")
        if effective_token:
            os.environ["HF_TOKEN"] = effective_token
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=effective_token,
                revision=revision,
            )
            log_message(
                f"Downloaded repository {repo_id} successfully.", verbose=verbose
            )
        except Exception as e:
            if target_dir.exists():
                target_resolved = target_dir.resolve()
                try:
                    target_resolved.relative_to(self.models_dir)
                    with contextlib.suppress(Exception):
                        shutil.rmtree(target_dir)
                except ValueError:
                    # target_dir is not within models/, skip deletion for safety
                    log_message(
                        f"Warning: Skipping deletion of {target_dir} as it is outside models/ directory",
                        always_print=True,
                    )
            raise ModelError(f"Failed to download repository {repo_id}: {e}") from e
        return target_dir

    def is_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            return model_type in self.models and self.models[model_type] is not None

    def load_upscale(self, verbose: bool = False):
        """Load upscale model (AnimeSharpV4 RCAN)."""
        with self._lock:
            if self.is_loaded(ModelType.UPSCALE):
                return self.models[ModelType.UPSCALE]

            log_message(
                "Loading upscale model (2x-AnimeSharpV4_RCAN)...", verbose=verbose
            )
            path = self.model_paths[ModelType.UPSCALE]

            # Try HF download first, fallback to direct URL
            try:
                hf_info = self.model_hf_repos[ModelType.UPSCALE]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                download_file(path, self.model_urls[ModelType.UPSCALE])

            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():  # noqa: SIM118
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    path, map_location=self.device, weights_only=False
                )

            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.UPSCALE] = model
            log_message("Upscale model loaded.", verbose=verbose)
            return model

    def load_upscale_lite(self, verbose: bool = False):
        """Load upscale lite model (AnimeSharpV4 Fast RCAN PU)."""
        with self._lock:
            if self.is_loaded(ModelType.UPSCALE_LITE):
                return self.models[ModelType.UPSCALE_LITE]

            log_message(
                "Loading upscale lite model (2x-AnimeSharpV4_Fast_RCAN_PU)...",
                verbose=verbose,
            )
            path = self.model_paths[ModelType.UPSCALE_LITE]

            # Try HF download first, fallback to direct URL
            try:
                hf_info = self.model_hf_repos[ModelType.UPSCALE_LITE]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                download_file(path, self.model_urls[ModelType.UPSCALE_LITE])

            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():  # noqa: SIM118
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    path, map_location=self.device, weights_only=False
                )

            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.UPSCALE_LITE] = model
            log_message("Upscale lite model loaded.", verbose=verbose)
            return model

    def _resolve_speech_bubble_model_type(self, model_path: str | None) -> ModelType:
        """Determine which speech bubble model type a path corresponds to."""
        if model_path is None:
            return ModelType.YOLO_SPEECH_BUBBLE
        # Callers may pass a path relative to the working directory
        p = Path(model_path).resolve()
        if p == self.model_paths[ModelType.YOLO_SPEECH_BUBBLE_2]:
            return ModelType.YOLO_SPEECH_BUBBLE_2
        return ModelType.YOLO_SPEECH_BUBBLE

    def load_yolo_speech_bubble(
        self, model_path: str | None = None, verbose: bool = False
    ):
        """Load YOLO model for speech bubble detection.

        Args:
            model_path: Optional custom path to YOLO model. If None, uses default.
            verbose: Whether to print verbose logging
        """
        with self._lock:
            model_type = self._resolve_speech_bubble_model_type(model_path)

            if self.is_loaded(model_type):
                return self.models[model_type]

            log_message(
                "Loading YOLO speech bubble detection model...", verbose=verbose
            )

            path = (
                self.model_paths[model_type]
                if model_path is None
                else Path(model_path).resolve()
            )

            if path == self.model_paths[model_type]:
                hf_info = self.model_hf_repos[model_type]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )

            model = YOLO(str(path))
            self.models[model_type] = model
            log_message("YOLO model loaded.", verbose=verbose)
            return model

    def load_rtdetr_conjoined_bubble(self, verbose: bool = False):
        """Load RT-DETR-v2 model for conjoined/fallback bubble detection.

        Returns a YOLO-compatible adapter exposing ``.names`` and a callable
        predict API used by the detection pipeline.
        """
        with self._lock:
            if self.is_loaded(ModelType.RTDETR_CONJOINED_BUBBLE):
                return self.models[ModelType.RTDETR_CONJOINED_BUBBLE]

            log_message(
                "Loading RT-DETR conjoined bubble detection model...", verbose=verbose
            )
            path = self.model_paths[ModelType.RTDETR_CONJOINED_BUBBLE]
            hf_info = self.model_hf_repos[ModelType.RTDETR_CONJOINED_BUBBLE]
            path = self._ensure_hf_repo(hf_info["repo_id"], path, verbose=verbose)

            try:
                processor = RTDetrImageProcessor.from_pretrained(str(path))
                model = RTDetrV2ForObjectDetection.from_pretrained(str(path))
                model.to(self.device)
                model.eval()
                adapter = RTDetrYOLOAdapter(
                    model=model,
                    processor=processor,
                    device=self.device,
                    names=getattr(model.config, "id2label", None),
                )
            except Exception as e:
                raise ModelError(f"Failed to load RT-DETR conjoined model: {e}") from e

            self.models[ModelType.RTDETR_CONJOINED_BUBBLE] = adapter
            log_message("RT-DETR conjoined bubble model loaded.", verbose=verbose)
            return adapter

    def load_yolo_osbtext(self, token: str | None = None, verbose: bool = False):
        """Load YOLO model for outside text detection.

        Args:
            token: Hugging Face token for gated repo access.
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.YOLO_OSBTEXT):
                return self.models[ModelType.YOLO_OSBTEXT]

            log_message("Loading YOLO OSB Text detection model...", verbose=verbose)

            path = self.model_paths[ModelType.YOLO_OSBTEXT]
            hf_info = self.model_hf_repos[ModelType.YOLO_OSBTEXT]

            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                path,
                token=token,
                verbose=verbose,
            )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_OSBTEXT] = model
            log_message("YOLO OSB Text model loaded.", verbose=verbose)
            return model

    def load_yolo_panel(self, verbose: bool = False):
        """Load YOLO model for panel detection.

        Args:
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.YOLO_PANEL):
                return self.models[ModelType.YOLO_PANEL]

            log_message("Loading YOLO panel detection model...", verbose=verbose)
            path = self.model_paths[ModelType.YOLO_PANEL]
            hf_info = self.model_hf_repos[ModelType.YOLO_PANEL]

            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                path,
                verbose=verbose,
            )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_PANEL] = model
            log_message("YOLO panel model loaded.", verbose=verbose)
            return model

    def load_manga_ocr(self, verbose: bool = False) -> Path:
        """Ensure manga-ocr model repository is downloaded.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded manga-ocr model directory
        """
        with self._lock:
            model_path = self.model_paths[ModelType.MANGA_OCR]
            hf_info = self.model_hf_repos[ModelType.MANGA_OCR]
            model_path = self._ensure_hf_repo(
                hf_info["repo_id"],
                model_path,
                revision=hf_info.get("revision"),
                verbose=verbose,
            )
            log_message("manga-ocr model repository ready.", verbose=verbose)
            return model_path

    def get_manga_ocr(self, verbose: bool = False):
        """Get manga-ocr instance, loading it if necessary.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            MangaOcr instance
        """
        with self._lock:
            if self.is_loaded(ModelType.MANGA_OCR):
                return self.models[ModelType.MANGA_OCR]

            log_message("Initializing manga-ocr...", verbose=verbose)

            # Fix for MeCab/Fugashi on non-Windows systems
            try:
                import os

                import unidic_lite

                os.environ["MECABRC"] = os.path.join(unidic_lite.DICDIR, "mecabrc")
            except ImportError:
                log_message(
                    "Warning: unidic_lite not found, skipping MeCab fix",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(f"Warning: Failed to apply MeCab fix: {e}", verbose=verbose)

            from manga_ocr import MangaOcr
            from transformers import logging as transformers_logging

            model_path = self.load_manga_ocr(verbose=verbose)

            # Suppress loading warnings (specifically for position_ids mismatch in transformers v5)
            previous_level = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                manga_ocr_instance = MangaOcr(
                    pretrained_model_name_or_path=str(model_path),
                    force_cpu=self.device.type == "cpu",
                )
            finally:
                transformers_logging.set_verbosity(previous_level)

            self.models[ModelType.MANGA_OCR] = manga_ocr_instance
            log_message("manga-ocr initialized", verbose=verbose)
            return manga_ocr_instance

    def load_paddle_ocr_vl(self, verbose: bool = False) -> Path:
        """Ensure PaddleOCR-VL-1.6 model repository is downloaded.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded PaddleOCR-VL-1.6 model directory
        """
        with self._lock:
            model_path = self.model_paths[ModelType.PADDLE_OCR_VL]
            hf_info = self.model_hf_repos[ModelType.PADDLE_OCR_VL]
            model_path = self._ensure_hf_repo(
                hf_info["repo_id"],
                model_path,
                revision=hf_info.get("revision"),
                verbose=verbose,
            )
            log_message("PaddleOCR-VL-1.6 model repository ready.", verbose=verbose)
            return model_path

    def get_paddle_ocr_vl(self, verbose: bool = False):
        """Get PaddleOCR-VL-1.6 (processor, model) tuple, loading if necessary.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Tuple of (AutoProcessor, AutoModelForImageTextToText)
        """
        with self._lock:
            if self.is_loaded(ModelType.PADDLE_OCR_VL):
                return self.models[ModelType.PADDLE_OCR_VL]

            log_message("Initializing PaddleOCR-VL-1.6...", verbose=verbose)

            from transformers import (
                AutoConfig,
                AutoModelForImageTextToText,
                AutoProcessor,
            )
            from transformers import logging as transformers_logging

            model_path = self.load_paddle_ocr_vl(verbose=verbose)
            model_path_str = str(model_path)
            token = self.hf_token or os.environ.get("HF_TOKEN")

            # Suppress upstream config, RoPE, and processor deprecation warnings from normal logs
            previous_level = None
            if not verbose:
                previous_level = transformers_logging.get_verbosity()
                transformers_logging.set_verbosity_error()
            try:
                config = AutoConfig.from_pretrained(model_path_str, token=token)
                config.tie_word_embeddings = False
                if hasattr(config, "text_config"):
                    config.text_config.tie_word_embeddings = False

                processor = AutoProcessor.from_pretrained(
                    model_path_str, token=token, backend="torchvision"
                )

                # Prefer flash_attention_2 on CUDA, fall back to sdpa on Windows/CPU
                dtype = self.dtype if self.device.type == "cuda" else None
                for attn_impl in ("flash_attention_2", "sdpa", "eager"):
                    try:
                        model = (
                            AutoModelForImageTextToText.from_pretrained(
                                model_path_str,
                                config=config,
                                dtype=dtype,
                                attn_implementation=attn_impl,
                                token=token,
                            )
                            .to(self.device)
                            .eval()
                        )
                        log_message(
                            f"PaddleOCR-VL-1.6 loaded with {attn_impl}", verbose=verbose
                        )
                        break
                    except (ImportError, ValueError, RuntimeError):
                        if attn_impl == "eager":
                            raise
                        log_message(
                            f"PaddleOCR-VL-1.6: {attn_impl} unavailable, trying fallback",
                            verbose=verbose,
                        )
            finally:
                if previous_level is not None:
                    transformers_logging.set_verbosity(previous_level)

            self.models[ModelType.PADDLE_OCR_VL] = (processor, model)
            log_message("PaddleOCR-VL-1.6 initialized.", verbose=verbose)
            return self.models[ModelType.PADDLE_OCR_VL]

    def load_sam2(self, verbose: bool = False):
        """Load SAM 2.1 model and processor.

        Returns:
            tuple: (processor, model) - SAM2 processor and model instances
        """
        with self._lock:
            if self.is_loaded(ModelType.SAM2):
                return self.models[ModelType.SAM2]

            log_message("Loading SAM 2.1 model...", verbose=verbose)
            hf_info = self.model_hf_repos[ModelType.SAM2]
            cache_dir = self._hf_cache_dir(self.sam_cache_dir)

            processor = Sam2Processor.from_pretrained(
                hf_info["repo_id"], cache_dir=cache_dir, token=self.hf_token
            )
            model = Sam2Model.from_pretrained(
                hf_info["repo_id"],
                torch_dtype=self.dtype,
                cache_dir=cache_dir,
                token=self.hf_token,
            ).to(self.device)
            model.eval()

            self.models[ModelType.SAM2] = (processor, model)
            log_message("SAM 2.1 model loaded.", verbose=verbose)
            return self.models[ModelType.SAM2]

    def load_sam3(self, token: str | None = None, verbose: bool = False):
        """Load SAM 3 Tracker (PVS) model and processor.

        SAM 3 requires a HuggingFace token with access to the gated facebook/sam3 repo.

        Args:
            token: HuggingFace token for gated model access
            verbose: Whether to print verbose logging

        Returns:
            tuple: (processor, model) - SAM3 Tracker processor and model instances
        """
        with self._lock:
            if self.is_loaded(ModelType.SAM3):
                return self.models[ModelType.SAM3]

            log_message("Loading SAM 3 Tracker model...", verbose=verbose)
            hf_info = self.model_hf_repos[ModelType.SAM3]
            cache_dir = self._hf_cache_dir(self.sam_cache_dir)

            effective_token = token if token is not None else self.hf_token
            processor = Sam3TrackerProcessor.from_pretrained(
                hf_info["repo_id"], cache_dir=cache_dir, token=effective_token
            )
            model = Sam3TrackerModel.from_pretrained(
                hf_info["repo_id"],
                torch_dtype=self.dtype,
                cache_dir=cache_dir,
                token=effective_token,
            ).to(self.device)
            model.eval()

            self.models[ModelType.SAM3] = (processor, model)
            log_message("SAM 3 Tracker model loaded.", verbose=verbose)
            return self.models[ModelType.SAM3]

    def set_device(self, device: torch.device):
        """Override the auto-detected device (e.g. --cpu). Call before loading models."""
        with self._lock:
            self.device = device
            self.dtype = get_best_dtype(device)

    def set_models_dir(self, models_dir: Path | str):
        """Store downloads under `models_dir` (e.g. --models). Call before loading models."""
        with self._lock:
            self._apply_models_dir(Path(models_dir))
            self.sdcpp_server_manager.install_dir = self.model_paths[
                ModelType.SDCPP_SERVER
            ]

    def set_hf_token(self, token: str, enable_fast_download: bool = True):
        """Set the global HuggingFace token for model downloads.

        Args:
            token: HuggingFace API token
            enable_fast_download: If True, enables HF_XET_HIGH_PERFORMANCE for faster downloads
        """
        with self._lock:
            self.hf_token = token if token else None
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
                if enable_fast_download:
                    self._enable_hf_xet_high_performance()
            elif "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]

    def set_flux_hf_token(self, token: str):
        """Set the HuggingFace token for Flux model downloads.

        Args:
            token: HuggingFace API token
        """
        with self._lock:
            self.flux_hf_token = token if token else None
            if self.flux_hf_token:
                os.environ["HF_TOKEN"] = self.flux_hf_token
            self._enable_hf_xet_high_performance()

    def set_flux_residual_diff_threshold(self, threshold: float):
        """Set the residual diff threshold for Flux caching.

        Args:
            threshold: Residual diff threshold (0.0-1.0)
        """
        self.flux_residual_diff_threshold = max(0.0, min(1.0, threshold))

    def load_flux_models(self, verbose: bool = False):
        """Load all Flux Kontext inpainting models (transformer, text encoder, pipeline).

        Returns:
            tuple: (transformer, text_encoder, pipeline)
        """
        with self._lock:
            if self.is_loaded(ModelType.FLUX_PIPELINE):
                return (
                    self.models[ModelType.FLUX_TRANSFORMER],
                    self.models[ModelType.FLUX_TEXT_ENCODER],
                    self.models[ModelType.FLUX_PIPELINE],
                )

            log_message("Loading Flux Kontext inpainting models...", verbose=verbose)
            try:
                # Lazy imports for Nunchaku and diffusers
                from diffusers import FluxKontextPipeline
                from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
                from nunchaku.models.text_encoders.t5_encoder import (
                    NunchakuT5EncoderModel,
                )
                from nunchaku.models.transformers.transformer_flux import (
                    NunchakuFluxTransformer2dModel,
                )
                from nunchaku.utils import get_precision

                hf_info = self.model_hf_repos[ModelType.FLUX_TRANSFORMER]
                if hf_info["filename"] is None:
                    hf_info["filename"] = (
                        f"svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
                    )
                transformer_path = self._ensure_hf_file(
                    hf_info["repo_id"],
                    hf_info["filename"],
                    self.diffusers_cache_dir / hf_info["filename"],
                    verbose=verbose,
                )
                transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                    str(transformer_path),
                    torch_dtype=self.dtype,
                    offload=True,
                    precision="int4",
                    set_attention_impl="nunchaku-fp16",
                )
                self.models[ModelType.FLUX_TRANSFORMER] = transformer

                hf_info = self.model_hf_repos[ModelType.FLUX_TEXT_ENCODER]
                text_encoder_path = self._ensure_hf_file(
                    hf_info["repo_id"],
                    hf_info["filename"],
                    self.diffusers_cache_dir / hf_info["filename"],
                    verbose=verbose,
                )
                text_encoder = NunchakuT5EncoderModel.from_pretrained(
                    str(text_encoder_path),
                    torch_dtype=self.dtype,
                )
                self.models[ModelType.FLUX_TEXT_ENCODER] = text_encoder

                pipeline_repo = self.model_hf_repos[ModelType.FLUX_PIPELINE]["repo_id"]
                effective_token = (
                    self.flux_hf_token if self.flux_hf_token else self.hf_token
                )
                pipeline = FluxKontextPipeline.from_pretrained(
                    pipeline_repo,
                    transformer=transformer,
                    text_encoder_2=text_encoder,
                    dtype=self.dtype,
                    cache_dir=self._hf_cache_dir(self.diffusers_cache_dir),
                    token=effective_token,
                ).to(self.device)

                apply_cache_on_pipe(
                    pipeline, residual_diff_threshold=self.flux_residual_diff_threshold
                )
                self.models[ModelType.FLUX_PIPELINE] = pipeline

                log_message("Flux Kontext models loaded successfully.", verbose=verbose)
                return transformer, text_encoder, pipeline
            except ImportError as e:
                raise ModelError(
                    "Nunchaku not installed or incompatible. Inpainting requires Nunchaku."
                ) from e
            except Exception as e:
                raise ModelError(
                    f"Failed to load Flux/Nunchaku inpainting models: {e}"
                ) from e

    def load_flux_kontext_sdnq(self, low_vram: bool = False, verbose: bool = False):
        """Load Flux.1 Kontext pipeline using SDNQ quantization (cross-platform).

        Unlike the Nunchaku method, this does not require a HuggingFace token
        and works on CUDA, ROCm, MPS, and CPU.

        Args:
            low_vram: If True, use sequential CPU offload (slower but lower VRAM)
            verbose: Whether to print verbose logging

        Returns:
            FluxKontextPipeline instance
        """
        with self._lock:
            if self.is_loaded(ModelType.FLUX_KONTEXT_SDNQ_PIPELINE):
                return self.models[ModelType.FLUX_KONTEXT_SDNQ_PIPELINE]

            log_message(
                "Loading Flux.1 Kontext SDNQ model (cross-platform)...", verbose=verbose
            )

            try:
                logging.getLogger("sdnq").setLevel(
                    logging.WARNING if verbose else logging.ERROR
                )
                from diffusers import FluxKontextPipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers
                from sdnq.common import use_torch_compile as triton_is_available
                from sdnq.loader import apply_sdnq_options_to_model

                hf_info = self.model_hf_repos[ModelType.FLUX_KONTEXT_SDNQ_PIPELINE]
                repo_id = hf_info["repo_id"]

                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                pipeline = FluxKontextPipeline.from_pretrained(
                    repo_id,
                    dtype=self.dtype,
                    cache_dir=self._hf_cache_dir(self.diffusers_cache_dir),
                )

                # Enable INT8 MatMul for GPU acceleration (AMD, Intel ARC, NVIDIA)
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch, "xpu") and torch.xpu.is_available()
                )
                use_quantized_matmul = triton_is_available and has_gpu
                if use_quantized_matmul:
                    log_message(
                        "Applying SDNQ INT8 MatMul optimization...", verbose=verbose
                    )
                pipeline.transformer = apply_sdnq_options_to_model(
                    pipeline.transformer,
                    dtype=self.dtype,
                    use_quantized_matmul=use_quantized_matmul,
                )
                pipeline.text_encoder_2 = apply_sdnq_options_to_model(
                    pipeline.text_encoder_2,
                    dtype=self.dtype,
                    use_quantized_matmul=use_quantized_matmul,
                )

                if low_vram:
                    log_message(
                        "Using sequential CPU offload (low VRAM mode)...",
                        verbose=verbose,
                    )
                    pipeline.enable_sequential_cpu_offload()
                else:
                    pipeline.enable_model_cpu_offload()

                self.models[ModelType.FLUX_KONTEXT_SDNQ_PIPELINE] = pipeline
                log_message(
                    "Flux.1 Kontext SDNQ model loaded successfully.",
                    verbose=verbose,
                )
                return pipeline

            except ImportError as e:
                raise ModelError(
                    "diffusers or sdnq not installed or incompatible. "
                    "Flux.1 Kontext SDNQ requires diffusers and sdnq."
                ) from e
            except Exception as e:
                raise ModelError(
                    f"Failed to load Flux.1 Kontext SDNQ model: {e}"
                ) from e

    def _load_flux_klein(
        self,
        model_type: ModelType,
        variant: str,
        low_vram: bool = False,
        verbose: bool = False,
    ):
        """Load a Flux.2 Klein model pipeline using SDNQ quantization.

        Both 4B and 9B use Disty0's SDNQ 4-bit quantized models.
        Supports AMD, Intel ARC, and NVIDIA GPUs with INT8 MatMul optimization.

        Args:
            model_type: ModelType.FLUX_KLEIN_9B_PIPELINE or FLUX_KLEIN_4B_PIPELINE
            variant: "9b" or "4b" for logging
            low_vram: If True, use sequential CPU offload (slower but lower VRAM)
            verbose: Whether to print verbose logging

        Returns:
            The loaded Flux2KleinPipeline
        """
        with self._lock:
            if self.is_loaded(model_type):
                return self.models[model_type]

            log_message(
                f"Loading Flux.2 Klein {variant.upper()} model...", verbose=verbose
            )

            try:
                logging.getLogger("sdnq").setLevel(
                    logging.WARNING if verbose else logging.ERROR
                )
                from diffusers import Flux2KleinPipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers
                from sdnq.common import use_torch_compile as triton_is_available
                from sdnq.loader import apply_sdnq_options_to_model

                hf_info = self.model_hf_repos[model_type]
                repo_id = hf_info["repo_id"]

                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                pipeline = Flux2KleinPipeline.from_pretrained(
                    repo_id,
                    dtype=self.dtype,
                    cache_dir=self._hf_cache_dir(self.diffusers_cache_dir),
                )

                # Enable INT8 MatMul for GPU acceleration (AMD, Intel ARC, NVIDIA)
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch, "xpu") and torch.xpu.is_available()
                )
                use_quantized_matmul = triton_is_available and has_gpu
                if use_quantized_matmul:
                    log_message(
                        "Applying SDNQ INT8 MatMul optimization...", verbose=verbose
                    )
                pipeline.transformer = apply_sdnq_options_to_model(
                    pipeline.transformer,
                    dtype=self.dtype,
                    use_quantized_matmul=use_quantized_matmul,
                )
                pipeline.text_encoder = apply_sdnq_options_to_model(
                    pipeline.text_encoder,
                    dtype=self.dtype,
                    use_quantized_matmul=use_quantized_matmul,
                )

                if low_vram:
                    log_message(
                        "Using sequential CPU offload (low VRAM mode)...",
                        verbose=verbose,
                    )
                    pipeline.enable_sequential_cpu_offload()
                else:
                    pipeline.enable_model_cpu_offload()

                self.models[model_type] = pipeline
                log_message(
                    f"Flux.2 Klein {variant.upper()} model loaded successfully.",
                    verbose=verbose,
                )
                return pipeline

            except ImportError as e:
                raise ModelError(
                    "diffusers or sdnq not installed or incompatible. Flux.2 Klein requires diffusers and sdnq."
                ) from e
            except Exception as e:
                raise ModelError(
                    f"Failed to load Flux.2 Klein {variant.upper()} model: {e}"
                ) from e

    def load_flux_klein_9b(self, low_vram: bool = False, verbose: bool = False):
        """Load Flux.2 Klein 9B pipeline with FP8 transformer.

        Note: Requires HuggingFace token with access to gated repo.

        Args:
            low_vram: If True, use sequential CPU offload (slower but lower VRAM)
            verbose: Whether to print verbose logging

        Returns:
            Flux2KleinPipeline instance
        """
        return self._load_flux_klein(
            ModelType.FLUX_KLEIN_9B_PIPELINE, "9b", low_vram=low_vram, verbose=verbose
        )

    def load_flux_klein_4b(self, low_vram: bool = False, verbose: bool = False):
        """Load Flux.2 Klein 4B pipeline with FP8 transformer.

        Args:
            low_vram: If True, use sequential CPU offload (slower but lower VRAM)
            verbose: Whether to print verbose logging

        Returns:
            Flux2KleinPipeline instance
        """
        return self._load_flux_klein(
            ModelType.FLUX_KLEIN_4B_PIPELINE, "4b", low_vram=low_vram, verbose=verbose
        )

    def _apply_sdnq_options(self, models, verbose: bool = False) -> None:
        """Enable SDNQ's INT8 MatMul on GPUs with Triton; sdnq updates the models in place."""
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        )
        use_quantized_matmul = triton_is_available and has_gpu
        if use_quantized_matmul:
            log_message("Applying SDNQ INT8 MatMul optimization...", verbose=verbose)
        for model in models:
            apply_sdnq_options_to_model(
                model, dtype=self.dtype, use_quantized_matmul=use_quantized_matmul
            )

    def _group_offload(self, *modules) -> None:
        """Leaf-level group offload, the lowest-VRAM diffusers option (streamed on CUDA)."""
        from diffusers.hooks import apply_group_offloading

        for module in modules:
            apply_group_offloading(
                module,
                onload_device=self.device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=self.device.type == "cuda",
            )

    def load_flux3_klein_sdnq(self, low_vram: bool = False, verbose: bool = False):
        """Load the Flux.3 Image Klein SDNQ 4-bit modular pipeline.

        Args:
            low_vram: If True, use leaf-level group offload (slower but lower VRAM)
            verbose: Whether to print verbose logging

        Returns:
            Flux3DistilledModularPipeline instance
        """
        with self._lock:
            if self.is_loaded(ModelType.FLUX3_KLEIN_SDNQ_PIPELINE):
                return self.models[ModelType.FLUX3_KLEIN_SDNQ_PIPELINE]

            log_message("Loading Flux.3 Image Klein SDNQ model...", verbose=verbose)
            try:
                logging.getLogger("sdnq").setLevel(
                    logging.WARNING if verbose else logging.ERROR
                )
                from diffusers import ComponentsManager, ModularPipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers

                repo_id = self.model_hf_repos[ModelType.FLUX3_KLEIN_SDNQ_PIPELINE][
                    "repo_id"
                ]
                cache_dir = self._hf_cache_dir(self.diffusers_cache_dir)
                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                blocks = ModularPipeline.from_pretrained(
                    repo_id, cache_dir=cache_dir
                ).blocks
                # The app sends its own prompt, so BFL's 9B prompt upsampler is never loaded
                blocks.sub_blocks.pop("prompt_upsampler")
                _add_cancel_check_to_denoise_loops(blocks)
                components_manager = None if low_vram else ComponentsManager()
                pipeline = blocks.init_pipeline(
                    repo_id, components_manager=components_manager
                )
                pipeline.load_components(dtype=self.dtype, cache_dir=cache_dir)
                # Flux.3 encodes text only (the reference goes through the VAE), so
                # Qwen3-VL's vision tower would just hold memory and, under group
                # offload, trigger diffusers' "layers were not executed" warning
                del pipeline.text_encoder.model.visual
                self._apply_sdnq_options(
                    (pipeline.transformer, pipeline.text_encoder), verbose=verbose
                )

                if low_vram:
                    log_message(
                        "Using group offload (low VRAM mode)...", verbose=verbose
                    )
                    self._group_offload(pipeline.transformer, pipeline.text_encoder)
                    pipeline.vae.to(self.device)
                else:
                    components_manager.enable_auto_cpu_offload(device=self.device)

                self.models[ModelType.FLUX3_KLEIN_SDNQ_PIPELINE] = pipeline
                log_message(
                    "Flux.3 Image Klein SDNQ model loaded successfully.",
                    verbose=verbose,
                )
                return pipeline

            except ImportError as e:
                raise ModelError(
                    "diffusers or sdnq not installed or incompatible. "
                    "Flux.3 Image Klein requires diffusers>=0.42.0 and sdnq>=0.2.8."
                ) from e
            except Exception as e:
                raise ModelError(
                    f"Failed to load Flux.3 Image Klein SDNQ model: {e}"
                ) from e

    def load_qwen_image_21_sdnq(self, low_vram: bool = False, verbose: bool = False):
        """Load the Qwen-Image 2.1 Turbo SDNQ 4-bit pipeline.

        Args:
            low_vram: If True, use leaf-level group offload and VAE tiling
            verbose: Whether to print verbose logging

        Returns:
            QwenImage21Pipeline instance
        """
        with self._lock:
            if self.is_loaded(ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE):
                return self.models[ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE]

            log_message("Loading Qwen-Image 2.1 Turbo SDNQ model...", verbose=verbose)
            try:
                logging.getLogger("sdnq").setLevel(
                    logging.WARNING if verbose else logging.ERROR
                )
                from diffusers import QwenImage21Pipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers

                repo_id = self.model_hf_repos[ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE][
                    "repo_id"
                ]
                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                pipeline = QwenImage21Pipeline.from_pretrained(
                    repo_id,
                    dtype=self.dtype,
                    cache_dir=self._hf_cache_dir(self.diffusers_cache_dir),
                )
                self._apply_sdnq_options(
                    (pipeline.transformer, pipeline.text_encoder), verbose=verbose
                )

                if low_vram:
                    log_message(
                        "Using group offload (low VRAM mode)...", verbose=verbose
                    )
                    self._group_offload(pipeline.transformer, pipeline.text_encoder)
                    pipeline.vae.to(self.device)
                    # An untiled ~1MP decode alone peaks above 8 GB
                    pipeline.vae.enable_tiling(
                        tile_sample_min_height=512, tile_sample_min_width=512
                    )
                else:
                    pipeline.enable_model_cpu_offload()

                self.models[ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE] = pipeline
                log_message(
                    "Qwen-Image 2.1 Turbo SDNQ model loaded successfully.",
                    verbose=verbose,
                )
                return pipeline

            except ImportError as e:
                raise ModelError(
                    "diffusers or sdnq not installed or incompatible. "
                    "Qwen-Image 2.1 Turbo requires diffusers>=0.41.0 and sdnq."
                ) from e
            except Exception as e:
                raise ModelError(
                    f"Failed to load Qwen-Image 2.1 Turbo SDNQ model: {e}"
                ) from e

    def load_lama_manga(self, verbose: bool = False):
        """Load the manga-finetuned Big-LaMa inpainting model through spandrel."""
        with self._lock:
            if self.is_loaded(ModelType.LAMA_MANGA):
                return self.models[ModelType.LAMA_MANGA]

            log_message("Loading LaMa manga inpainting model...", verbose=verbose)
            from safetensors.torch import load_file

            hf_info = self.model_hf_repos[ModelType.LAMA_MANGA]
            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                self.model_paths[ModelType.LAMA_MANGA],
                verbose=verbose,
            )
            # The checkpoint dropped the "generator." prefix that spandrel detects LaMa by
            state_dict = {
                f"generator.{key}": tensor
                for key, tensor in load_file(str(path)).items()
            }
            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.LAMA_MANGA] = model
            log_message("LaMa manga inpainting model loaded.", verbose=verbose)
            return model

    def load_text_sam(self, verbose: bool = False):
        """Load Koharu's manga-tuned SAM-TS-L, which segments text pixels on a page."""
        with self._lock:
            if self.is_loaded(ModelType.TEXT_SAM):
                # A no-op unless park_text_sam moved it to RAM
                self._text_sam_parked = False
                return self.models[ModelType.TEXT_SAM].to(self.device)

            log_message("Loading SAM-TS-L text segmentation model...", verbose=verbose)
            from safetensors.torch import load_file

            from core.ml.hi_sam import build_text_sam_ts_l, to_inference_device

            hf_info = self.model_hf_repos[ModelType.TEXT_SAM]
            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                self.model_paths[ModelType.TEXT_SAM],
                verbose=verbose,
            )
            model = build_text_sam_ts_l()
            model.load_state_dict(load_file(str(path)), strict=True)
            model = to_inference_device(model, self.device).eval()
            self.models[ModelType.TEXT_SAM] = model
            self._text_sam_parked = False
            log_message("SAM-TS-L text segmentation model loaded.", verbose=verbose)
            return model

    def unload_model(
        self, model_type: ModelType, force_gc: bool = True, verbose: bool = False
    ):
        """Unload a specific model and free memory.

        Args:
            model_type: Type of model to unload
            force_gc: Whether to force garbage collection
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if not self.is_loaded(model_type):
                return

            log_message(f"Unloading {model_type.value}...", verbose=verbose)
            del self.models[model_type]
            self.models[model_type] = None

            if force_gc:
                empty_cache(self.device)

    def _unload_models(self, model_types, verbose: bool = False) -> bool:
        """Unload each loaded model in `model_types`; True if any was loaded."""
        loaded = [
            model_type for model_type in model_types if self.is_loaded(model_type)
        ]
        for model_type in loaded:
            self.unload_model(model_type, force_gc=False, verbose=verbose)
        if loaded:
            empty_cache(self.device)
        return bool(loaded)

    def unload_upscale_models(self, verbose: bool = False):
        """Unload upscale models (both regular and lite)."""
        self._unload_models((ModelType.UPSCALE, ModelType.UPSCALE_LITE), verbose)
        log_message("Upscale models unloaded.", verbose=verbose)

    def unload_ocr_models(self, verbose: bool = False):
        """Unload OCR-related models (YOLO, SAM2/SAM3, and manga-ocr)."""
        ocr_models = (
            ModelType.YOLO_SPEECH_BUBBLE,
            ModelType.YOLO_SPEECH_BUBBLE_2,
            ModelType.RTDETR_CONJOINED_BUBBLE,
            ModelType.SAM2,
            ModelType.SAM3,
            ModelType.YOLO_OSBTEXT,
            ModelType.YOLO_PANEL,
            ModelType.MANGA_OCR,
            ModelType.PADDLE_OCR_VL,
        )
        if self._unload_models(ocr_models, verbose):
            log_message("OCR models unloaded.", verbose=verbose)

    def unload_flux_kontext_models(self, verbose: bool = False):
        """Unload all Flux.1 Kontext models."""
        unloaded = self._unload_models(
            (
                ModelType.FLUX_TRANSFORMER,
                ModelType.FLUX_TEXT_ENCODER,
                ModelType.FLUX_PIPELINE,
            ),
            verbose,
        )
        self.shutdown_sdcpp_server("flux_kontext", verbose=verbose)
        if unloaded:
            log_message("Flux.1 Kontext models unloaded.", verbose=verbose)

    def unload_flux_kontext_sdnq_models(self, verbose: bool = False):
        """Unload Flux.1 Kontext SDNQ model."""
        if self._unload_models((ModelType.FLUX_KONTEXT_SDNQ_PIPELINE,), verbose):
            log_message("Flux.1 Kontext SDNQ model unloaded.", verbose=verbose)

    def unload_flux_klein_models(self, verbose: bool = False):
        """Unload all Flux.2 Klein models."""
        unloaded = self._unload_models(
            (ModelType.FLUX_KLEIN_9B_PIPELINE, ModelType.FLUX_KLEIN_4B_PIPELINE),
            verbose,
        )
        self.shutdown_sdcpp_server("flux_klein_9b", verbose=verbose)
        self.shutdown_sdcpp_server("flux_klein_4b", verbose=verbose)
        if unloaded:
            log_message("Flux.2 Klein models unloaded.", verbose=verbose)

    def unload_flux3_klein_models(self, verbose: bool = False):
        """Unload Flux.3 Image Klein (SDNQ pipeline and sd.cpp server)."""
        unloaded = self._unload_models((ModelType.FLUX3_KLEIN_SDNQ_PIPELINE,), verbose)
        self.shutdown_sdcpp_server("flux3_klein", verbose=verbose)
        if unloaded:
            log_message("Flux.3 Image Klein model unloaded.", verbose=verbose)

    def unload_qwen_image_21_models(self, verbose: bool = False):
        """Unload Qwen-Image 2.1 Turbo (SDNQ pipeline and sd.cpp server)."""
        unloaded = self._unload_models(
            (ModelType.QWEN_IMAGE_21_SDNQ_PIPELINE,), verbose
        )
        self.shutdown_sdcpp_server("qwen_image_2_1_turbo", verbose=verbose)
        if unloaded:
            log_message("Qwen-Image 2.1 Turbo model unloaded.", verbose=verbose)

    def unload_lama_manga(self, verbose: bool = False):
        """Unload the LaMa manga inpainting model and the text segmenter it uses."""
        if self._unload_models((ModelType.LAMA_MANGA, ModelType.TEXT_SAM), verbose):
            log_message("LaMa manga inpainting model unloaded.", verbose=verbose)

    def park_text_sam(self, verbose: bool = False):
        """Free the SAM-TS-L text model's VRAM for a diffusion inpainter, keeping it in
        RAM (~0.7 GB) for the next page; a no-op once parked.

        Moving it back takes ~0.15 s on CUDA, against ~2.2 s to reload it from disk. On
        the CPU it is unloaded instead, since there it would stay in fp32 (~1.4 GB) while
        the diffusion inpainter needs the RAM.
        """
        if self.device.type == "cpu":
            if self._unload_models((ModelType.TEXT_SAM,), verbose):
                log_message(
                    "SAM-TS-L text segmentation model unloaded.", verbose=verbose
                )
            return
        with self._lock:
            if not self.is_loaded(ModelType.TEXT_SAM) or self._text_sam_parked:
                return
            self.models[ModelType.TEXT_SAM].to("cpu")
            self._text_sam_parked = True
            empty_cache(self.device)
        log_message("SAM-TS-L text segmentation model moved to RAM.", verbose=verbose)

    def unload_all(self, verbose: bool = False):
        """Unload all models and free all GPU memory."""
        log_message("Unloading all models...", verbose=verbose)
        self.shutdown_sdcpp_servers(verbose=verbose)
        with self._lock:
            for model_type in list(self.models.keys()):
                if self.is_loaded(model_type):
                    del self.models[model_type]
                    self.models[model_type] = None

            empty_cache(self.device)
            log_message("All models unloaded.", verbose=verbose)

    def get_memory_stats(self):
        """Get current GPU memory usage statistics."""
        return get_device_info(self.device)

    def print_memory_stats(self):
        """Print current GPU memory usage."""
        stats = self.get_memory_stats()
        if stats["memory"] == "N/A":
            log_message(f"Device: {stats['device']}", always_print=True)
        else:
            log_message(
                f"GPU Memory - Allocated: {stats['allocated_gb']} GB, "
                f"Reserved: {stats['reserved_gb']} GB",
                always_print=True,
            )

    def clear_cache(self):
        """Release unused GPU memory from PyTorch's CUDA cache."""
        empty_cache(self.device)


_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
