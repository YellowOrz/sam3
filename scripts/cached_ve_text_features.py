"""Opt-in fixed-hand VE feature cache, optionally with a zero-initialized delta.

This is not a general text encoder and is not enabled by any existing builder.
The complete original VE triple is retained. Four valid positions are updated
in ``zero_delta``; ``content_delta`` updates only the middle two (body text),
without deleting or masking the start/end features. CPU feature equality alone
is not full SAM3 validation.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from typing import Sequence

import torch
from torch import nn


CLASS_NAMES = ("left_hand", "right_hand")
NATURAL_PROMPTS = ("left hand", "right hand")
CACHE_FORMAT = "sam3-cached-natural-ve-hand-features-v1"


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def expected_delta_shape(mode: str) -> tuple[int, int, int]:
    """Trainable output-delta layout; a frozen cache has no delta layout."""
    if mode == "zero_delta":
        return (2, 4, 256)
    if mode == "content_delta":
        return (2, 2, 256)
    raise ValueError("Delta mode must be 'zero_delta' or 'content_delta'")


def _validate_tensors(padding: torch.Tensor, resized: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    for value, shape, name in ((padding, (2, 32), "padding"), (resized, (32, 2, 256), "resized"),
                               (raw, (32, 2, 1024), "raw")):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if padding.dtype != torch.bool:
        raise ValueError("padding must be bool, True means padding/ignored")
    if not resized.is_floating_point() or not raw.is_floating_point():
        raise ValueError("resized and raw features must be floating tensors")
    if len({padding.device, resized.device, raw.device}) != 1:
        raise ValueError("All cache tensors must be on one explicit device")
    if padding.device.type == "meta":
        raise ValueError("Materialized features are required, not meta tensors")
    if not bool(torch.isfinite(resized).all()) or not bool(torch.isfinite(raw).all()):
        raise ValueError("Cached VE features must be finite, including padded positions")
    positions = [(~row).nonzero(as_tuple=False).flatten() for row in padding]
    if any(len(row) != 4 for row in positions):
        raise ValueError("Each natural hand prompt must have exactly four valid positions; refusing to pool or guess")
    return torch.stack(positions)


def _metadata(value: dict, *, resized, raw, positions) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Cache metadata must be a JSON dictionary")
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("Cache metadata must contain finite JSON-compatible values") from error
    for name in ("base_checkpoint_sha256", "tokenizer_sha256"):
        if not _valid_sha(result.get(name)):
            raise ValueError(f"Required cache provenance SHA256: {name}")
    expected = {"format": CACHE_FORMAT, "class_names": list(CLASS_NAMES),
                "prompt_texts": list(NATURAL_PROMPTS), "context_length": 32,
                "resized_dimension": 256, "raw_dimension": 1024,
                "padding_dtype": "torch.bool", "resized_dtype": str(resized.dtype),
                "raw_dtype": str(raw.dtype), "valid_positions": positions.cpu().tolist()}
    for name, expected_value in expected.items():
        if name in result and result[name] != expected_value:
            raise ValueError(f"Cache metadata disagrees with features or fixed prompts: {name}")
        result[name] = expected_value
    return result


def validate_delta_state(state: Mapping) -> tuple[int, int, int]:
    """Validate a complete trainable cache state, without mutation or RNG use.

    Returns the expected FP32 delta shape. Frozen/unknown modes are rejected;
    callers loading a frozen initializer must explicitly construct their chosen
    zero-initialized delta mode first. Content positions are the second and
    third entries of each class's four ordered non-padding positions, not two
    replacement tokens or a different padding mask.
    """
    fields = {"delta", "padding_cache", "resized_cache", "raw_cache", "valid_positions", "_extra_state"}
    if not isinstance(state, Mapping) or set(state) != fields:
        raise ValueError("A complete trainable CachedVETextEncoder state is required")
    extra = state["_extra_state"]
    if not isinstance(extra, dict) or set(extra) != {"mode", "metadata"}:
        raise ValueError("Delta state mode/metadata is malformed")
    shape = expected_delta_shape(extra["mode"])
    positions = _validate_tensors(state["padding_cache"], state["resized_cache"], state["raw_cache"])
    saved_positions = state["valid_positions"]
    if (not isinstance(saved_positions, torch.Tensor) or saved_positions.dtype != torch.int64
            or saved_positions.device != positions.device or tuple(saved_positions.shape) != (2, 4)
            or not torch.equal(saved_positions, positions)):
        raise ValueError("Delta state valid positions disagree with the padding mask")
    delta = state["delta"]
    if (not isinstance(delta, torch.Tensor) or tuple(delta.shape) != shape
            or delta.dtype != torch.float32 or delta.device != positions.device
            or not bool(torch.isfinite(delta).all())):
        raise ValueError("Delta state must have the mode-specific finite FP32 shape/device")
    _metadata(extra["metadata"], resized=state["resized_cache"], raw=state["raw_cache"], positions=positions)
    return shape


class CachedVETextEncoder(nn.Module):
    """Return fixed natural-prompt VE features for internal left/right class keys.

    ``frozen`` has no parameters. ``zero_delta`` has exactly 2*4*256 FP32
    parameters added only to valid resized-feature positions. ``content_delta``
    has 2*2*256 FP32 parameters, added only at the middle two valid positions;
    start/end output features stay frozen but continue participating downstream.
    The 32-position padding mask and raw 1024-dimensional embeddings never change.

    Move explicitly with ``encoder.to(device=...)``. Changing feature dtype is
    rejected at forward time because it invalidates cached-VE equivalence.
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(self, padding_mask: torch.Tensor, resized_features: torch.Tensor,
                 raw_features: torch.Tensor, *, metadata: dict, mode: str = "frozen"):
        super().__init__()
        if mode not in ("frozen", "zero_delta", "content_delta"):
            raise ValueError("mode must be 'frozen', 'zero_delta' or 'content_delta'")
        positions = _validate_tensors(padding_mask, resized_features, raw_features)
        self.mode = mode
        self.context_length = 32
        self.d_model = 256
        self.class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        self._cache_metadata = _metadata(metadata, resized=resized_features, raw=raw_features, positions=positions)
        # A caller may capture VE under inference_mode. Materialize ordinary
        # detached tensors so a subsequent delta backward/optimizer remains legal.
        with torch.inference_mode(False):
            self.register_buffer("padding_cache", padding_mask.detach().clone())
            self.register_buffer("resized_cache", resized_features.detach().clone())
            self.register_buffer("raw_cache", raw_features.detach().clone())
            self.register_buffer("valid_positions", positions.detach().clone())
            if mode != "frozen":
                self.delta = nn.Parameter(torch.zeros(expected_delta_shape(mode), dtype=torch.float32,
                                                     device=resized_features.device))
            else:
                self.register_parameter("delta", None)

    @property
    def cache_metadata(self) -> dict:
        return deepcopy(self._cache_metadata)

    def get_extra_state(self):
        return {"mode": self.mode, "metadata": self.cache_metadata}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("mode") != self.mode:
            raise ValueError("Cannot load a cache/delta state into a different mode")
        positions = _validate_tensors(self.padding_cache, self.resized_cache, self.raw_cache)
        if not torch.equal(positions, self.valid_positions):
            raise ValueError("Loaded delta positions disagree with the cached padding mask")
        if self.delta is not None:
            validate_delta_state({"delta": self.delta, "padding_cache": self.padding_cache,
                "resized_cache": self.resized_cache, "raw_cache": self.raw_cache,
                "valid_positions": self.valid_positions, "_extra_state": state})
        self._cache_metadata = _metadata(state.get("metadata"), resized=self.resized_cache,
                                         raw=self.raw_cache, positions=positions)

    def forward(self, captions: Sequence[str], input_boxes=None, device=None):
        if isinstance(captions, (str, bytes)) or not isinstance(captions, (list, tuple)) or not captions:
            raise ValueError("captions must be a nonempty list/tuple of fixed hand class keys")
        unknown = [caption for caption in captions if not isinstance(caption, str) or caption not in self.class_to_index]
        if unknown:
            raise ValueError(f"Unknown fixed hand class {unknown[0]!r}; expected exactly {CLASS_NAMES}")
        if input_boxes is not None and len(input_boxes) > 0:
            raise ValueError("Cached hand VE features do not support input boxes or placeholders")
        if device is not None and torch.device(device) != self.resized_cache.device:
            requested = torch.device(device)
            # torch.device('cuda') is the current CUDA device, not a distinct device.
            matches_current_cuda = (requested.type == "cuda" and requested.index is None
                                    and self.resized_cache.device.type == "cuda"
                                    and self.resized_cache.device.index == torch.cuda.current_device())
            if not matches_current_cuda:
                raise ValueError("Move CachedVETextEncoder explicitly with .to(device=...) before requesting another device")
        if (str(self.resized_cache.dtype) != self._cache_metadata["resized_dtype"]
                or str(self.raw_cache.dtype) != self._cache_metadata["raw_dtype"]
                or self.padding_cache.dtype != torch.bool):
            raise ValueError("Cache dtype changed; exact original VE dtypes must be preserved")
        if self.delta is not None and self.delta.dtype != torch.float32:
            raise ValueError("Delta parameters must stay FP32; use autocast rather than module dtype conversion")
        if self.delta is not None and tuple(self.delta.shape) != expected_delta_shape(self.mode):
            raise ValueError("Delta parameter shape disagrees with its declared mode")
        indices = torch.tensor([self.class_to_index[caption] for caption in captions],
                               dtype=torch.long, device=self.resized_cache.device)
        padding = self.padding_cache.index_select(0, indices)
        resized = self.resized_cache.index_select(1, indices)
        raw = self.raw_cache.index_select(1, indices)
        if self.delta is not None:
            positions = self.valid_positions.index_select(0, indices)
            if self.mode == "content_delta":
                positions = positions[:, 1:3]
            rows = torch.arange(len(captions), device=indices.device)[:, None].expand(-1, positions.shape[1])
            delta = self.delta.index_select(0, indices).to(dtype=resized.dtype)
            # Do not shortcut delta==0: it must have gradients at initialization.
            # Only these valid positions are written, leaving padded bytes untouched.
            resized[positions, rows] = resized[positions, rows] + delta
        return padding, resized, raw


def capture_ve_text_cache(ve_encoder: nn.Module, *, base_checkpoint_sha256: str,
                          tokenizer_sha256: str, device=None, mode: str = "frozen",
                          metadata: dict | None = None) -> CachedVETextEncoder:
    """Capture the *actual* VE triple once under the caller's precision context.

    This intentionally does not change the original encoder's mode or precision.
    Caller must select the same autocast policy used for later equivalence tests.
    Hash arguments identify previously verified files; this helper does not hash
    model weights or assert that a caller-provided hash matches a particular file.
    """
    if ve_encoder.training or any(module.training for module in ve_encoder.modules()):
        raise ValueError("Capture requires the original VE encoder and all submodules in eval mode")
    if isinstance(ve_encoder, CachedVETextEncoder):
        raise ValueError("Capture requires the original VE, not an existing cache")
    if not _valid_sha(base_checkpoint_sha256) or not _valid_sha(tokenizer_sha256):
        raise ValueError("Capture requires valid base/tokenizer SHA256 provenance")
    details = dict(metadata or {})
    for name, value in (("base_checkpoint_sha256", base_checkpoint_sha256), ("tokenizer_sha256", tokenizer_sha256)):
        if name in details and details[name] != value:
            raise ValueError(f"Conflicting capture metadata: {name}")
        details[name] = value
    details.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    details.setdefault("equivalence_scope", "text feature cache only; full SAM3 GPU equivalence requires a separate check")
    details["capture_encoder_class"] = f"{type(ve_encoder).__module__}.{type(ve_encoder).__qualname__}"
    with torch.no_grad():
        result = ve_encoder(list(NATURAL_PROMPTS), input_boxes=None, device=device)
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError("Original VE must return exactly (padding, resized, raw)")
    return CachedVETextEncoder(*result, metadata=details, mode=mode)


def _installed_encoder(model: nn.Module) -> CachedVETextEncoder:
    encoder = getattr(getattr(model, "backbone", None), "language_backbone", None)
    if not isinstance(encoder, CachedVETextEncoder):
        raise ValueError("No explicitly installed CachedVETextEncoder at model.backbone.language_backbone")
    return encoder


def set_cached_ve_training_mode(model: nn.Module, *, train_delta: bool) -> set[str]:
    """Keep frozen SAM3 in eval mode while optionally enabling delta gradients.

    ``eval`` disables dropout but does not disable autograd. Use this instead of
    ``model.train()`` for the frozen-base delta experiment.
    """
    encoder = _installed_encoder(model)
    if train_delta and encoder.delta is None:
        raise ValueError("Frozen cache has no delta parameters to train")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if encoder.delta is not None:
        encoder.delta.requires_grad_(train_delta)
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def _require_frozen_base_eval(model, _inputs):
    if any(module.training for module in model.modules()):
        raise RuntimeError("Cached-VE frozen-base path requires eval mode; use set_cached_ve_training_mode, not model.train()")


def install_cached_ve_text_encoder(model: nn.Module, encoder: CachedVETextEncoder) -> nn.Module:
    """Explicitly replace only the language backbone, return the original VE.

    No model is rebuilt and no vision/decoder tensor is reinitialized. The
    original encoder is returned for a same-base comparison or restoration.
    Installation freezes existing weights and enables only a present delta.
    """
    if not isinstance(encoder, CachedVETextEncoder):
        raise TypeError("encoder must be an explicitly constructed CachedVETextEncoder")
    backbone = getattr(model, "backbone", None)
    previous = getattr(backbone, "language_backbone", None)
    if not isinstance(previous, nn.Module):
        raise ValueError("Expected an existing model.backbone.language_backbone")
    if isinstance(previous, CachedVETextEncoder) or hasattr(model, "_cached_ve_eval_guard_handle"):
        raise ValueError("A cached VE encoder is already installed; refusing an implicit replacement")
    for parameter in previous.parameters():
        parameter.requires_grad_(False)
    previous.eval()
    backbone.language_backbone = encoder
    set_cached_ve_training_mode(model, train_delta=encoder.delta is not None)
    model._cached_ve_eval_guard_handle = model.register_forward_pre_hook(_require_frozen_base_eval)
    return previous


def restore_original_ve_text_encoder(model: nn.Module, original_encoder: nn.Module) -> CachedVETextEncoder:
    """Remove this opt-in guard and restore the caller-retained original VE.

    The model stays in eval mode and its base parameters stay frozen; this is
    for controlled forward comparisons, not implicit resumption of full tuning.
    """
    cached = _installed_encoder(model)
    if not isinstance(original_encoder, nn.Module) or isinstance(original_encoder, CachedVETextEncoder):
        raise ValueError("Provide the original noncached VE module returned by install")
    handle = getattr(model, "_cached_ve_eval_guard_handle", None)
    if handle is None:
        raise ValueError("Missing explicit cached-VE installation guard")
    handle.remove()
    delattr(model, "_cached_ve_eval_guard_handle")
    model.backbone.language_backbone = original_encoder
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return cached
