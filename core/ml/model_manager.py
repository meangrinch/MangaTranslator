import gc
import shutil
import threading
import urllib.request
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import easyocr
import torch
from diffusers import FluxKontextPipeline
from huggingface_hub import hf_hub_download
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
from spandrel import ModelLoader
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO

from utils.exceptions import ModelError
from utils.logging import log_message


class ModelType(Enum):
    """Enumeration of available model types."""

    UPSCALE = "upscale"
    YOLO_SPEECH_BUBBLE = "yolo_speech_bubble"
    YOLO_CONJOINED_BUBBLE = "yolo_conjoined_bubble"
    EASYOCR = "easyocr"
    SAM2 = "sam2"
    FLUX_TRANSFORMER = "flux_transformer"
    FLUX_TEXT_ENCODER = "flux_text_encoder"
    FLUX_PIPELINE = "flux_pipeline"


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

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            )

            # Model storage
            self.models = {}
            self.model_paths = self._init_model_paths()
            self.model_urls = self._init_model_urls()
            self.model_hf_repos = self._init_hf_repos()

            # Flux-specific configuration
            self.flux_cache_dir = Path("./models/flux")
            # Note: Users should set their own Hugging Face token if needed
            self.flux_hf_token = None
            self.flux_residual_diff_threshold = 0.12

            self._initialized = True
            log_message(
                f"Model Manager initialized on device: {self.device}", always_print=True
            )

    def _init_model_paths(self):
        """Initialize model file paths."""
        model_dir = Path("./models").resolve()
        return {
            ModelType.UPSCALE: (
                model_dir / "upscale" / "2x-AnimeSharpV4_RCAN.safetensors"
            ),
            ModelType.YOLO_SPEECH_BUBBLE: (
                model_dir / "yolo" / "yolov8m_seg-speech-bubble.pt"
            ),
            ModelType.YOLO_CONJOINED_BUBBLE: (
                model_dir / "yolo" / "comic-speech-bubble-detector-yolov8m.pt"
            ),
        }

    def _init_model_urls(self):
        """Initialize model download URLs."""
        return {
            ModelType.UPSCALE: (
                "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
                "2x-AnimeSharpV4_RCAN.safetensors"
            ),
        }

    def _init_hf_repos(self):
        """Initialize Hugging Face repository information."""
        repos = {
            ModelType.UPSCALE: {
                "repo_id": "Kim2091/2x-AnimeSharpV4",
                "filename": "2x-AnimeSharpV4_RCAN.safetensors",
            },
            ModelType.YOLO_SPEECH_BUBBLE: {
                "repo_id": "kitsumed/yolov8m_seg-speech-bubble",
                "filename": "model.pt",
            },
            ModelType.YOLO_CONJOINED_BUBBLE: {
                "repo_id": "ogkalu/comic-speech-bubble-detector-yolov8m",
                "filename": "comic-speech-bubble-detector.pt",
            },
            ModelType.SAM2: {
                "repo_id": "facebook/sam2.1-hiera-large",
            },
            ModelType.FLUX_PIPELINE: {
                "repo_id": "black-forest-labs/FLUX.1-Kontext-dev",
                "filename": None,  # Pipeline loaded via from_pretrained
            },
        }

        repos[ModelType.FLUX_TRANSFORMER] = {
            "repo_id": "nunchaku-tech/nunchaku-flux.1-kontext-dev",
            "filename": f"svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors",
        }
        repos[ModelType.FLUX_TEXT_ENCODER] = {
            "repo_id": "nunchaku-tech/nunchaku-t5",
            "filename": "awq-int4-flux.1-t5xxl.safetensors",
        }

        return repos

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
        log_message(f"Downloading {path.name}...", verbose=verbose)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(path, "wb") as f:
                shutil.copyfileobj(response, f)
            log_message(f"Downloaded {path.name} successfully.", verbose=verbose)
        except Exception as e:
            if path.exists():
                path.unlink()
            raise ModelError(f"Failed to download {path.name}: {e}")

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
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        log_message(
            f"Downloading {target.name} from Hugging Face ({repo_id})...",
            verbose=verbose,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target.parent),
            token=token,
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
        log_message(f"Downloaded {target.name} successfully.", verbose=verbose)
        return target

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

    def load_yolo_speech_bubble(
        self, model_path: Optional[str] = None, verbose: bool = False
    ):
        """Load YOLO model for speech bubble detection.

        Args:
            model_path: Optional custom path to YOLO model. If None, uses default.
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.YOLO_SPEECH_BUBBLE):
                return self.models[ModelType.YOLO_SPEECH_BUBBLE]

            log_message(
                "Loading YOLO speech bubble detection model...", verbose=verbose
            )

            path = (
                self.model_paths[ModelType.YOLO_SPEECH_BUBBLE]
                if model_path is None
                else Path(model_path)
            )

            if path == self.model_paths[ModelType.YOLO_SPEECH_BUBBLE]:
                hf_info = self.model_hf_repos[ModelType.YOLO_SPEECH_BUBBLE]
                self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_SPEECH_BUBBLE] = model
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

    def load_easyocr(self, languages=None, verbose: bool = False):
        """Load EasyOCR reader.

        Args:
            languages: List of language codes. Defaults to ["ja"].
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.EASYOCR):
                return self.models[ModelType.EASYOCR]

            if languages is None:
                languages = ["ja"]

            log_message(
                f"Loading EasyOCR reader for languages: {languages}...",
                verbose=verbose,
            )
            reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
            self.models[ModelType.EASYOCR] = reader
            log_message("EasyOCR reader loaded.", verbose=verbose)
            return reader

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

            processor = Sam2Processor.from_pretrained(hf_info["repo_id"], cache_dir=cache_dir)
            model = Sam2Model.from_pretrained(
                hf_info["repo_id"], torch_dtype=self.dtype, cache_dir=cache_dir
            ).to(self.device)
            model.eval()

            # Store as tuple
            self.models[ModelType.SAM2] = (processor, model)
            log_message("SAM 2.1 model loaded.", verbose=verbose)
            return self.models[ModelType.SAM2]

    def set_flux_hf_token(self, token: str):
        """Set the HuggingFace token for Flux model downloads.

        Args:
            token: HuggingFace API token
        """
        self.flux_hf_token = token if token else None

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
                # Load transformer
                hf_info = self.model_hf_repos[ModelType.FLUX_TRANSFORMER]
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
                pipeline = FluxKontextPipeline.from_pretrained(
                    pipeline_repo,
                    transformer=transformer,
                    text_encoder_2=text_encoder,
                    torch_dtype=self.dtype,
                    cache_dir=str(self.flux_cache_dir),
                    token=self.flux_hf_token,
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

            if force_gc and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    def unload_upscale_models(self, verbose: bool = False):
        """Unload upscale model."""
        self.unload_model(ModelType.UPSCALE, force_gc=True, verbose=verbose)
        log_message("Upscale model unloaded.", verbose=verbose)

    def unload_ocr_models(self, verbose: bool = False):
        """Unload OCR-related models (YOLO, SAM2, and EasyOCR)."""
        models_unloaded = []
        if self.is_loaded(ModelType.YOLO_SPEECH_BUBBLE):
            models_unloaded.append("yolo_speech_bubble")
        if self.is_loaded(ModelType.YOLO_CONJOINED_BUBBLE):
            models_unloaded.append("yolo_conjoined_bubble")
        if self.is_loaded(ModelType.SAM2):
            models_unloaded.append("sam2")
        if self.is_loaded(ModelType.EASYOCR):
            models_unloaded.append("easyocr")

        self.unload_model(ModelType.YOLO_SPEECH_BUBBLE, force_gc=False, verbose=verbose)
        self.unload_model(
            ModelType.YOLO_CONJOINED_BUBBLE, force_gc=False, verbose=verbose
        )
        self.unload_model(ModelType.SAM2, force_gc=False, verbose=verbose)
        self.unload_model(ModelType.EASYOCR, force_gc=True, verbose=verbose)

        if models_unloaded:
            log_message("OCR models unloaded.", verbose=verbose)

    def unload_flux_models(self, verbose: bool = False):
        """Unload all Flux Kontext models."""
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

        if models_unloaded:
            log_message("Flux Kontext models unloaded.", verbose=verbose)

    def unload_all(self, verbose: bool = False):
        """Unload all models and free all GPU memory."""
        with self._lock:
            log_message("Unloading all models...", verbose=verbose)
            for model_type in list(self.models.keys()):
                if self.is_loaded(model_type):
                    del self.models[model_type]
                    self.models[model_type] = None

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            log_message("All models unloaded.", verbose=verbose)

    def get_memory_stats(self):
        """Get current GPU memory usage statistics."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory": "N/A"}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": f"{allocated:.2f}",
            "reserved_gb": f"{reserved:.2f}",
        }

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

    @contextmanager
    def upscale_context(self, verbose: bool = False):
        """Context manager for upscale model - auto-loads and unloads."""
        try:
            self.load_upscale(verbose=verbose)
            yield self.models[ModelType.UPSCALE]
        finally:
            self.unload_upscale_models(verbose=verbose)

    @contextmanager
    def ocr_context(self, languages=None, verbose: bool = False):
        """Context manager for OCR models - auto-loads and unloads.

        Args:
            languages: List of language codes for EasyOCR
            verbose: Whether to print verbose logging
        """
        try:
            yolo = self.load_yolo_speech_bubble(verbose=verbose)
            easyocr_reader = self.load_easyocr(languages, verbose=verbose)
            yield yolo, easyocr_reader
        finally:
            self.unload_ocr_models(verbose=verbose)

    @contextmanager
    def flux_context(self, verbose: bool = False):
        """Context manager for Flux models - auto-loads and unloads."""
        try:
            transformer, text_encoder, pipeline = self.load_flux_models(verbose=verbose)
            yield transformer, text_encoder, pipeline
        finally:
            self.unload_flux_models(verbose=verbose)


# Global singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
