"""Validate and load the token-only checkpoints produced by token training."""

from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

import torch

from sam3.model.learnable_text_encoder import LearnableClassTextEncoder


def load_class_token_checkpoint(
    path: Union[str, Path], *, tokens_per_class: Optional[int] = None
) -> torch.Tensor:
    """Load validated CPU tokens, inferring K from the tensor rather than defaults."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Token checkpoint must contain a mapping")
    class_names = state.get("class_names")
    expected_names = LearnableClassTextEncoder.CLASS_NAMES
    if not isinstance(class_names, (list, tuple)) or tuple(class_names) != expected_names:
        raise ValueError(
            "Token checkpoint class_names must be ordered as "
            f"{list(LearnableClassTextEncoder.CLASS_NAMES)}, got {class_names!r}"
        )

    tokens = state.get("class_tokens")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[0] != len(LearnableClassTextEncoder.CLASS_NAMES)
        or tokens.shape[1] < 1
        or tokens.shape[2] != 256
    ):
        raise ValueError(
            "Token checkpoint class_tokens must have shape (2, K, 256), K >= 1"
        )
    if tokens.layout != torch.strided or not tokens.is_floating_point():
        raise ValueError(
            "Token checkpoint class_tokens must be a dense floating-point tensor"
        )
    if not torch.isfinite(tokens).all().item():
        raise ValueError("Token checkpoint class_tokens contains non-finite values")

    inferred_k = int(tokens.shape[1])
    for name, expected in (("tokens_per_class", inferred_k), ("d_model", 256)):
        if name in state and (type(state[name]) is not int or state[name] != expected):
            raise ValueError(
                f"Token checkpoint {name}={state[name]!r} disagrees with "
                f"class_tokens shape {tuple(tokens.shape)}"
            )
    if tokens_per_class is not None and (
        type(tokens_per_class) is not int or tokens_per_class != inferred_k
    ):
        raise ValueError(
            f"--tokens-per-class={tokens_per_class!r} conflicts with "
            f"token checkpoint K={inferred_k}"
        )
    return tokens.detach()


def copy_class_tokens(
    encoder: LearnableClassTextEncoder, tokens: torch.Tensor
) -> None:
    """Overlay a validated token tensor without touching the frozen base weights."""
    if not isinstance(encoder, LearnableClassTextEncoder):
        raise ValueError("Detector language encoder must be LearnableClassTextEncoder")
    if tuple(encoder.class_tokens.shape) != tuple(tokens.shape):
        raise ValueError(
            f"Detector tokens shape {tuple(encoder.class_tokens.shape)} does not "
            f"match checkpoint shape {tuple(tokens.shape)}"
        )
    with torch.no_grad():
        converted_tokens = tokens.to(encoder.class_tokens)
        if not torch.isfinite(converted_tokens).all().item():
            raise ValueError("Token values are non-finite in the detector's dtype")
        encoder.class_tokens.copy_(converted_tokens)
