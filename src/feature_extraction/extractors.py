from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

    HAS_TRANSFORMERS = True
except Exception:  # noqa: BLE001
    CLIPImageProcessor = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]
    CLIPTokenizerFast = None  # type: ignore[assignment]
    HAS_TRANSFORMERS = False

try:
    import clip as openai_clip
except Exception:  # noqa: BLE001
    openai_clip = None

from .config import EncoderConfig, RuntimeConfig


def set_deterministic_mode(enabled: bool, seed: int) -> None:
    if not enabled:
        return
    # Required by CuBLAS on CUDA >= 10.2 for deterministic matmul kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        if device.type != "cuda":
            raise ValueError("float16 requires CUDA device")
        return torch.float16
    raise ValueError(f"Unsupported dtype `{dtype_name}`")


def l2_normalize(features: np.ndarray, epsilon: float) -> np.ndarray:
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)
    return features / norms


@dataclass(frozen=True)
class ExtractorInfo:
    embedding_dim: int
    device: str
    dtype: str
    backend: str


class ClipFeatureExtractor:
    def __init__(self, encoder_cfg: EncoderConfig, runtime_cfg: RuntimeConfig) -> None:
        self.encoder_cfg = encoder_cfg
        self.runtime_cfg = runtime_cfg
        set_deterministic_mode(runtime_cfg.deterministic, runtime_cfg.seed)

        self.device = resolve_torch_device(runtime_cfg.device)
        self.dtype = resolve_torch_dtype(runtime_cfg.dtype, self.device)
        self.cache_dir = runtime_cfg.huggingface_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = encoder_cfg.text_max_length
        self.normalize_epsilon = encoder_cfg.normalize_epsilon
        self.backend = ""

        last_error: Exception | None = None
        if HAS_TRANSFORMERS:
            try:
                self._init_transformers_backend()
                self.backend = "transformers"
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if self.backend == "" and openai_clip is not None:
            self._init_openai_clip_backend()
            self.backend = "openai_clip"

        if self.backend == "":
            if last_error is not None:
                raise RuntimeError(
                    "Failed to initialize CLIP backend. transformers backend failed and openai clip "
                    "package is unavailable."
                ) from last_error
            raise RuntimeError("No available CLIP backend (transformers or clip package).")

    def _init_transformers_backend(self) -> None:
        assert CLIPModel is not None
        assert CLIPImageProcessor is not None
        assert CLIPTokenizerFast is not None
        self.model = CLIPModel.from_pretrained(self.encoder_cfg.model_name, cache_dir=str(self.cache_dir))
        self.model.to(self.device)
        self.model.eval()

        image_proc_cfg = self.encoder_cfg.image_preprocess
        crop_size = int(image_proc_cfg.get("crop_size", 224))
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.encoder_cfg.model_name,
            cache_dir=str(self.cache_dir),
            do_resize=True,
            size={"shortest_edge": int(image_proc_cfg.get("resize_shortest", 224))},
            resample=Image.BICUBIC,
            do_center_crop=True,
            crop_size={"height": crop_size, "width": crop_size},
            do_convert_rgb=True,
            do_normalize=True,
        )
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            self.encoder_cfg.model_name,
            cache_dir=str(self.cache_dir),
        )

    def _init_openai_clip_backend(self) -> None:
        if openai_clip is None:
            raise RuntimeError("openai clip package is not installed.")
        model_name = self._map_model_name_for_openai_clip(self.encoder_cfg.model_name)
        model, preprocess = openai_clip.load(
            model_name,
            device=str(self.device),
            jit=False,
            download_root=str(self.cache_dir),
        )
        model.eval()
        if self.dtype == torch.float16:
            model = model.half()
        self.model = model
        self.openai_clip_preprocess = preprocess

    @staticmethod
    def _map_model_name_for_openai_clip(model_name: str) -> str:
        mapping = {
            "openai/clip-vit-base-patch32": "ViT-B/32",
            "ViT-B/32": "ViT-B/32",
        }
        if model_name not in mapping:
            raise ValueError(f"Unsupported model_name for openai clip backend: {model_name}")
        return mapping[model_name]

    def info(self) -> ExtractorInfo:
        if self.backend == "transformers":
            embedding_dim = int(self.model.config.projection_dim)
        else:
            text_projection = getattr(self.model, "text_projection", None)
            if text_projection is None:
                raise RuntimeError("openai clip model missing text_projection")
            embedding_dim = int(text_projection.shape[-1])
        return ExtractorInfo(
            embedding_dim=embedding_dim,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            backend=self.backend,
        )

    def encode_images(self, absolute_image_paths: Sequence[Path]) -> np.ndarray:
        if self.backend == "transformers":
            images = []
            for path in absolute_image_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.dtype == torch.float16:
                pixel_values = pixel_values.half()
            with torch.inference_mode():
                outputs = self.model.get_image_features(pixel_values=pixel_values)
            features = outputs.detach().cpu().float().numpy()
        else:
            tensors = []
            for path in absolute_image_paths:
                with Image.open(path) as img:
                    tensors.append(self.openai_clip_preprocess(img.convert("RGB")))
            image_input = torch.stack(tensors, dim=0).to(self.device)
            if self.dtype == torch.float16:
                image_input = image_input.half()
            with torch.inference_mode():
                outputs = self.model.encode_image(image_input)
            features = outputs.detach().cpu().float().numpy()
        if self.encoder_cfg.normalize_type == "l2":
            features = l2_normalize(features, epsilon=self.normalize_epsilon)
        return features.astype(np.float32, copy=False)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [text if text is not None else "" for text in texts]
        if self.backend == "transformers":
            tokenized = self.tokenizer(
                normalized_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            with torch.inference_mode():
                outputs = self.model.get_text_features(**tokenized)
            features = outputs.detach().cpu().float().numpy()
        else:
            tokens = openai_clip.tokenize(normalized_texts, context_length=self.max_length, truncate=True).to(
                self.device
            )
            with torch.inference_mode():
                outputs = self.model.encode_text(tokens)
            features = outputs.detach().cpu().float().numpy()
        if self.encoder_cfg.normalize_type == "l2":
            features = l2_normalize(features, epsilon=self.normalize_epsilon)
        return features.astype(np.float32, copy=False)
