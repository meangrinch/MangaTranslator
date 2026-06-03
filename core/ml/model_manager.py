import os
import shutil
import threading
import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from spandrel import ModelLoader
from transformers import (
    Sam2Model,
    Sam2Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)
from ultralytics import YOLO

from core.device import empty_cache, get_best_device, get_best_dtype, get_device_info
from core.ml.sdcpp_server import SDCppServerManager
from utils.exceptions import ModelError
from utils.logging import log_message
from utils.model_metadata import FLUX_SDCPP_QUANT_FILES


class ModelType(Enum):
    """Enumeration of available model types."""

    UPSCALE = "upscale"
    UPSCALE_LITE = "upscale_lite"
    YOLO_SPEECH_BUBBLE = "yolo_speech_bubble"
    YOLO_SPEECH_BUBBLE_2 = "yolo_speech_bubble_2"
    YOLO_CONJOINED_BUBBLE = "yolo_conjoined_bubble"
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
    SDCPP_SERVER = "sdcpp_server"
    FLUX_KLEIN_SDCPP_VAE = "flux_klein_sdcpp_vae"
    FLUX_KONTEXT_SDCPP_CLIP_L = "flux_kontext_sdcpp_clip_l"
    FLUX_KONTEXT_SDCPP_VAE = "flux_kontext_sdcpp_vae"


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

            # Model storage
            self.models = {}
            self.model_paths = self._init_model_paths()
            self.model_urls = self._init_model_urls()
            self.model_hf_repos = self._init_hf_repos()

            # Flux-specific configuration
            self.flux_cache_dir = Path("./models/flux")
            self.hf_token = None
            self.flux_hf_token = None
            self.flux_residual_diff_threshold = 0.15
            self.sdcpp_server_manager = SDCppServerManager(
                self.model_paths[ModelType.SDCPP_SERVER]
            )

            # Serializes Flux pipeline inference across threads (CPU offload is not thread-safe)
            self.flux_inference_lock = threading.Lock()

            self._initialized = True
            log_message(
                f"Model Manager initialized on device: {self.device}", always_print=True
            )

    @staticmethod
    def _flux_sdcpp_base_dir() -> Path:
        """Base directory for sd.cpp Flux GGUF assets."""
        return Path("./models").resolve() / "flux" / "sdcpp"

    def _init_model_paths(self):
        """Initialize model file paths."""
        model_dir = Path("./models").resolve()
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
                model_dir / "yolo" / "yolov8m_seg-speech-bubble.pt"
            ),
            ModelType.YOLO_SPEECH_BUBBLE_2: (
                model_dir / "yolo" / "manga109-segmentation-bubble.pt"
            ),
            ModelType.YOLO_CONJOINED_BUBBLE: (
                model_dir / "yolo" / "comic-speech-bubble-detector-yolov8m.pt"
            ),
            ModelType.YOLO_OSBTEXT: (model_dir / "yolo" / "animetext_yolov12x.pt"),
            ModelType.YOLO_PANEL: (
                model_dir / "yolo" / "manga109_v2023.12.07_l_yolov11.pt"
            ),
            ModelType.MANGA_OCR: (model_dir / "manga-ocr-base"),
            ModelType.PADDLE_OCR_VL: (model_dir / "paddleocr-vl"),
            ModelType.SDCPP_SERVER: (
                model_dir / "sdcpp" / SDCppServerManager.RELEASE_TAG
            ),
            ModelType.FLUX_KLEIN_SDCPP_VAE: (flux_sdcpp_dir / "flux2-vae.safetensors"),
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
            ModelType.FLUX_KLEIN_SDCPP_VAE: (
                "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/"
                "split_files/vae/flux2-vae.safetensors"
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
            ModelType.YOLO_CONJOINED_BUBBLE: {
                "repo_id": "ogkalu/comic-speech-bubble-detector-yolov8m",
                "filename": "comic-speech-bubble-detector.pt",
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
            "repo_id": "PaddlePaddle/PaddleOCR-VL-1.5",
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
        repos[ModelType.FLUX_KLEIN_SDCPP_VAE] = {
            "repo_id": "Comfy-Org/flux2-dev",
            "filename": "split_files/vae/flux2-vae.safetensors",
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

    def _ensure_file(self, path: Path, url: str, verbose: bool = False) -> None:
        """Download file from URL if it doesn't exist.

        Args:
            path: Path where file should be saved
            url: URL to download from
            verbose: Whether to print verbose logging
        """
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        log_message(f"Downloading {path.name}...", always_print=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(path, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                last_log = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_log >= 5:
                        if total:
                            percent = min(100.0, downloaded * 100.0 / total)
                            log_message(
                                f"  - {path.name}: {downloaded / (1024**2):.1f}/"
                                f"{total / (1024**2):.1f} MiB ({percent:.1f}%)",
                                always_print=True,
                            )
                        else:
                            log_message(
                                f"  - {path.name}: {downloaded / (1024**2):.1f} MiB",
                                always_print=True,
                            )
                        last_log = now
            log_message(f"Downloaded {path.name} successfully.", always_print=True)
        except Exception as e:
            if path.exists():
                path.unlink()
            raise ModelError(f"Failed to download {path.name}: {e}")

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
        return {
            "repo_id": quant_spec["repo_id"],
            "filename": filename,
            "path": base_dir / filename,
            "quant": selected_quant,
        }

    def _flux_sdcpp_asset_spec(
        self,
        model_key: str,
        diffusion_quant: str = "",
        text_encoder_quant: str = "",
    ) -> dict:
        specs = {
            "flux_klein_4b": {
                "label": "Flux.2 Klein 4B",
                "quant_models": {
                    "diffusion_model": diffusion_quant,
                    "llm": text_encoder_quant,
                },
                "fixed_model_types": {
                    "vae": ModelType.FLUX_KLEIN_SDCPP_VAE,
                },
                "server_args": ("--llm", "llm"),
            },
            "flux_klein_9b": {
                "label": "Flux.2 Klein 9B",
                "quant_models": {
                    "diffusion_model": diffusion_quant,
                    "llm": text_encoder_quant,
                },
                "fixed_model_types": {
                    "vae": ModelType.FLUX_KLEIN_SDCPP_VAE,
                },
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
        }
        if model_key not in specs:
            raise ModelError(f"Unknown sd.cpp Flux model key: {model_key}")
        spec = specs[model_key]
        spec["quant_assets"] = {
            key: self._flux_sdcpp_quant_asset(model_key, key, quant)
            for key, quant in spec["quant_models"].items()
        }
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
                "server_model_key": f"{model_key}-{server_quant_key}",
            }
            for key, asset in spec["quant_assets"].items():
                assets[key] = asset["path"]
                self._ensure_hf_file(
                    asset["repo_id"],
                    asset["filename"],
                    assets[key],
                    token=self.flux_hf_token,
                    verbose=verbose,
                )

            for key, model_type in spec["fixed_model_types"].items():
                assets[key] = self.model_paths[model_type]
                hf_info = self.model_hf_repos[model_type]
                self._ensure_hf_file(
                    hf_info["repo_id"],
                    hf_info["filename"],
                    assets[key],
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
        token: Optional[str] = None,
        verbose: bool = False,
    ) -> Path:
        """Download file from Hugging Face if it doesn't exist.

        Args:
            repo_id: Hugging Face repository ID
            filename: Name of file to download
            target: Path where file should be saved
            token: Optional Hugging Face token
            verbose: Whether to print verbose logging
        """
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
                try:
                    downloaded_path.unlink()
                except Exception:
                    pass

            self._remove_empty_child_dirs(target.parent, verbose=verbose)
        log_message(f"Downloaded {target.name} successfully.", verbose=verbose)
        return target

    def _ensure_hf_repo(
        self,
        repo_id: str,
        target_dir: Path,
        token: Optional[str] = None,
        revision: Optional[str] = None,
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
        # Check for safetensors for transformers v5+
        safetensors_file = target_dir / "model.safetensors"
        bin_file = target_dir / "pytorch_model.bin"

        is_downloaded = False
        if target_dir.exists():
            if safetensors_file.exists():
                is_downloaded = True
            elif bin_file.exists():
                if revision is None:
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
                models_dir = Path("./models").resolve()
                target_resolved = target_dir.resolve()
                try:
                    target_resolved.relative_to(models_dir)
                    # Safe to delete
                    try:
                        shutil.rmtree(target_dir)
                    except Exception:
                        pass
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
                self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                self._ensure_file(
                    path, self.model_urls[ModelType.UPSCALE], verbose=verbose
                )

            # Load model
            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():
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
                self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                self._ensure_file(
                    path, self.model_urls[ModelType.UPSCALE_LITE], verbose=verbose
                )

            # Load model
            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():
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

    def _resolve_speech_bubble_model_type(self, model_path: Optional[str]) -> ModelType:
        """Determine which speech bubble model type a path corresponds to."""
        if model_path is None:
            return ModelType.YOLO_SPEECH_BUBBLE
        p = Path(model_path)
        if p == self.model_paths[ModelType.YOLO_SPEECH_BUBBLE_2]:
            return ModelType.YOLO_SPEECH_BUBBLE_2
        return ModelType.YOLO_SPEECH_BUBBLE

    def load_yolo_speech_bubble(
        self, model_path: Optional[str] = None, verbose: bool = False
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
                self.model_paths[model_type] if model_path is None else Path(model_path)
            )

            if path == self.model_paths[model_type]:
                hf_info = self.model_hf_repos[model_type]
                self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )

            model = YOLO(str(path))
            self.models[model_type] = model
            log_message("YOLO model loaded.", verbose=verbose)
            return model

    def load_yolo_conjoined_bubble(self, verbose: bool = False):
        """Load YOLO model for conjoined speech bubble detection."""
        with self._lock:
            if self.is_loaded(ModelType.YOLO_CONJOINED_BUBBLE):
                return self.models[ModelType.YOLO_CONJOINED_BUBBLE]

            log_message(
                "Loading YOLO conjoined bubble detection model...", verbose=verbose
            )
            path = self.model_paths[ModelType.YOLO_CONJOINED_BUBBLE]

            # Try HF download
            hf_info = self.model_hf_repos[ModelType.YOLO_CONJOINED_BUBBLE]
            self._ensure_hf_file(
                hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
            )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_CONJOINED_BUBBLE] = model
            log_message("YOLO conjoined bubble model loaded.", verbose=verbose)
            return model

    def load_yolo_osbtext(self, token: Optional[str] = None, verbose: bool = False):
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

            self._ensure_hf_file(
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

            self._ensure_hf_file(
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
            self._ensure_hf_repo(
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

            # Ensure model is downloaded
            model_path = self.load_manga_ocr(verbose=verbose)

            # Suppress loading warnings (specifically for position_ids mismatch in transformers v5)
            previous_level = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                manga_ocr_instance = MangaOcr(
                    pretrained_model_name_or_path=str(model_path)
                )
            finally:
                transformers_logging.set_verbosity(previous_level)

            self.models[ModelType.MANGA_OCR] = manga_ocr_instance
            log_message("manga-ocr initialized", verbose=verbose)
            return manga_ocr_instance

    def load_paddle_ocr_vl(self, verbose: bool = False) -> Path:
        """Ensure PaddleOCR-VL-1.5 model repository is downloaded.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded PaddleOCR-VL-1.5 model directory
        """
        with self._lock:
            model_path = self.model_paths[ModelType.PADDLE_OCR_VL]
            hf_info = self.model_hf_repos[ModelType.PADDLE_OCR_VL]
            self._ensure_hf_repo(
                hf_info["repo_id"],
                model_path,
                revision=hf_info.get("revision"),
                verbose=verbose,
            )
            log_message("PaddleOCR-VL-1.5 model repository ready.", verbose=verbose)
            return model_path

    def get_paddle_ocr_vl(self, verbose: bool = False):
        """Get PaddleOCR-VL-1.5 (processor, model) tuple, loading if necessary.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Tuple of (AutoProcessor, AutoModelForImageTextToText)
        """
        with self._lock:
            if self.is_loaded(ModelType.PADDLE_OCR_VL):
                return self.models[ModelType.PADDLE_OCR_VL]

            log_message("Initializing PaddleOCR-VL-1.5...", verbose=verbose)

            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_path = self.load_paddle_ocr_vl(verbose=verbose)
            model_path_str = str(model_path)
            token = self.hf_token or os.environ.get("HF_TOKEN")

            processor = AutoProcessor.from_pretrained(
                model_path_str, token=token, use_fast=True
            )

            # Prefer flash_attention_2 on CUDA, fall back to sdpa on Windows/CPU
            dtype = self.dtype if self.device.type == "cuda" else None
            for attn_impl in ("flash_attention_2", "sdpa", "eager"):
                try:
                    model = (
                        AutoModelForImageTextToText.from_pretrained(
                            model_path_str,
                            torch_dtype=dtype,
                            attn_implementation=attn_impl,
                            token=token,
                        )
                        .to(self.device)
                        .eval()
                    )
                    log_message(
                        f"PaddleOCR-VL-1.5 loaded with {attn_impl}", verbose=verbose
                    )
                    break
                except (ImportError, ValueError, RuntimeError):
                    if attn_impl == "eager":
                        raise
                    log_message(
                        f"PaddleOCR-VL-1.5: {attn_impl} unavailable, trying fallback",
                        verbose=verbose,
                    )

            self.models[ModelType.PADDLE_OCR_VL] = (processor, model)
            log_message("PaddleOCR-VL-1.5 initialized.", verbose=verbose)
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
            cache_dir = "models/sam"

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

            # Store as tuple
            self.models[ModelType.SAM2] = (processor, model)
            log_message("SAM 2.1 model loaded.", verbose=verbose)
            return self.models[ModelType.SAM2]

    def load_sam3(self, token: Optional[str] = None, verbose: bool = False):
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
            cache_dir = "models/sam"

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
                    self.flux_cache_dir / hf_info["filename"],
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

                # Load text encoder
                hf_info = self.model_hf_repos[ModelType.FLUX_TEXT_ENCODER]
                text_encoder_path = self._ensure_hf_file(
                    hf_info["repo_id"],
                    hf_info["filename"],
                    self.flux_cache_dir / hf_info["filename"],
                    verbose=verbose,
                )
                text_encoder = NunchakuT5EncoderModel.from_pretrained(
                    str(text_encoder_path),
                    torch_dtype=self.dtype,
                )
                self.models[ModelType.FLUX_TEXT_ENCODER] = text_encoder

                # Load pipeline
                pipeline_repo = self.model_hf_repos[ModelType.FLUX_PIPELINE]["repo_id"]
                effective_token = (
                    self.flux_hf_token if self.flux_hf_token else self.hf_token
                )
                pipeline = FluxKontextPipeline.from_pretrained(
                    pipeline_repo,
                    transformer=transformer,
                    text_encoder_2=text_encoder,
                    torch_dtype=self.dtype,
                    cache_dir=str(self.flux_cache_dir),
                    token=effective_token,
                ).to(self.device)

                # Apply caching for faster inference
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
                from diffusers import FluxKontextPipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers
                from sdnq.common import use_torch_compile as triton_is_available
                from sdnq.loader import apply_sdnq_options_to_model

                hf_info = self.model_hf_repos[ModelType.FLUX_KONTEXT_SDNQ_PIPELINE]
                repo_id = hf_info["repo_id"]

                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                pipeline = FluxKontextPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=self.dtype,
                    cache_dir=str(self.flux_cache_dir),
                )

                # Enable INT8 MatMul for GPU acceleration (AMD, Intel ARC, NVIDIA)
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch, "xpu") and torch.xpu.is_available()
                )
                if triton_is_available and has_gpu:
                    log_message(
                        "Applying SDNQ INT8 MatMul optimization...", verbose=verbose
                    )
                    pipeline.transformer = apply_sdnq_options_to_model(
                        pipeline.transformer, use_quantized_matmul=True
                    )
                    pipeline.text_encoder_2 = apply_sdnq_options_to_model(
                        pipeline.text_encoder_2, use_quantized_matmul=True
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
                from diffusers import Flux2KleinPipeline
                from sdnq import SDNQConfig  # noqa: F401 - registers into diffusers
                from sdnq.common import use_torch_compile as triton_is_available
                from sdnq.loader import apply_sdnq_options_to_model

                hf_info = self.model_hf_repos[model_type]
                repo_id = hf_info["repo_id"]

                log_message(f"Loading SDNQ pipeline from {repo_id}...", verbose=verbose)
                pipeline = Flux2KleinPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=self.dtype,
                    cache_dir=str(self.flux_cache_dir),
                )

                # Enable INT8 MatMul for GPU acceleration (AMD, Intel ARC, NVIDIA)
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch, "xpu") and torch.xpu.is_available()
                )
                if triton_is_available and has_gpu:
                    log_message(
                        "Applying SDNQ INT8 MatMul optimization...", verbose=verbose
                    )
                    pipeline.transformer = apply_sdnq_options_to_model(
                        pipeline.transformer, use_quantized_matmul=True
                    )
                    pipeline.text_encoder = apply_sdnq_options_to_model(
                        pipeline.text_encoder, use_quantized_matmul=True
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

    def unload_upscale_models(self, verbose: bool = False):
        """Unload upscale models (both regular and lite)."""
        self.unload_model(ModelType.UPSCALE, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.UPSCALE_LITE, force_gc=False, verbose=verbose)
        empty_cache(self.device)
        log_message("Upscale models unloaded.", verbose=verbose)

    def unload_ocr_models(self, verbose: bool = False):
        """Unload OCR-related models (YOLO, SAM2/SAM3, and manga-ocr)."""
        models_unloaded = []
        if self.is_loaded(ModelType.YOLO_SPEECH_BUBBLE):
            models_unloaded.append("yolo_speech_bubble")
        if self.is_loaded(ModelType.YOLO_SPEECH_BUBBLE_2):
            models_unloaded.append("yolo_speech_bubble_2")
        if self.is_loaded(ModelType.YOLO_CONJOINED_BUBBLE):
            models_unloaded.append("yolo_conjoined_bubble")
        if self.is_loaded(ModelType.SAM2):
            models_unloaded.append("sam2")
        if self.is_loaded(ModelType.SAM3):
            models_unloaded.append("sam3")
        if self.is_loaded(ModelType.YOLO_OSBTEXT):
            models_unloaded.append("yolo_osbtext")
        if self.is_loaded(ModelType.YOLO_PANEL):
            models_unloaded.append("yolo_panel")
        if self.is_loaded(ModelType.MANGA_OCR):
            models_unloaded.append("manga_ocr")
        if self.is_loaded(ModelType.PADDLE_OCR_VL):
            models_unloaded.append("paddle_ocr_vl")

        self.unload_model(ModelType.YOLO_SPEECH_BUBBLE, force_gc=False, verbose=verbose)
        self.unload_model(
            ModelType.YOLO_SPEECH_BUBBLE_2, force_gc=False, verbose=verbose
        )
        self.unload_model(
            ModelType.YOLO_CONJOINED_BUBBLE, force_gc=False, verbose=verbose
        )
        self.unload_model(ModelType.SAM2, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.SAM3, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.YOLO_OSBTEXT, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.YOLO_PANEL, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.MANGA_OCR, force_gc=True, verbose=verbose)
        self.unload_model(ModelType.PADDLE_OCR_VL, force_gc=True, verbose=verbose)

        if models_unloaded:
            log_message("OCR models unloaded.", verbose=verbose)

    def unload_flux_kontext_models(self, verbose: bool = False):
        """Unload all Flux.1 Kontext models."""
        models_unloaded = []
        if self.is_loaded(ModelType.FLUX_TRANSFORMER):
            models_unloaded.append("flux_transformer")
        if self.is_loaded(ModelType.FLUX_TEXT_ENCODER):
            models_unloaded.append("flux_text_encoder")
        if self.is_loaded(ModelType.FLUX_PIPELINE):
            models_unloaded.append("flux_pipeline")

        self.unload_model(ModelType.FLUX_TRANSFORMER, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.FLUX_TEXT_ENCODER, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.FLUX_PIPELINE, force_gc=True, verbose=verbose)
        self.shutdown_sdcpp_server("flux_kontext", verbose=verbose)

        if models_unloaded:
            log_message("Flux.1 Kontext models unloaded.", verbose=verbose)

    def unload_flux_kontext_sdnq_models(self, verbose: bool = False):
        """Unload Flux.1 Kontext SDNQ model."""
        if self.is_loaded(ModelType.FLUX_KONTEXT_SDNQ_PIPELINE):
            self.unload_model(
                ModelType.FLUX_KONTEXT_SDNQ_PIPELINE, force_gc=True, verbose=verbose
            )
            log_message("Flux.1 Kontext SDNQ model unloaded.", verbose=verbose)

    def unload_flux_klein_models(self, verbose: bool = False):
        """Unload all Flux.2 Klein models."""
        models_unloaded = []
        if self.is_loaded(ModelType.FLUX_KLEIN_9B_PIPELINE):
            models_unloaded.append("flux_klein_9b")
        if self.is_loaded(ModelType.FLUX_KLEIN_4B_PIPELINE):
            models_unloaded.append("flux_klein_4b")

        self.unload_model(
            ModelType.FLUX_KLEIN_9B_PIPELINE, force_gc=False, verbose=verbose
        )
        self.unload_model(
            ModelType.FLUX_KLEIN_4B_PIPELINE, force_gc=True, verbose=verbose
        )
        self.shutdown_sdcpp_server("flux_klein_9b", verbose=verbose)
        self.shutdown_sdcpp_server("flux_klein_4b", verbose=verbose)

        if models_unloaded:
            log_message("Flux.2 Klein models unloaded.", verbose=verbose)

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


# Global singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
