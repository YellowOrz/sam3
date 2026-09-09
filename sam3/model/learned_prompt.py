"""One independently trained target, replacing the projected text token sequence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LearnedPrompt(nn.Module):
    """Only valid tokens are parameters; padding stays fixed, even with weight decay."""

    def __init__(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        target_id: str,
        metadata: dict | None = None,
    ) -> None:
        super().__init__()
        if features.shape != (32, 256) or not features.is_floating_point():
            raise ValueError("features must be a floating point [32, 256] tensor")
        if mask.shape != (32,) or mask.dtype != torch.bool or mask.all():
            raise ValueError("mask must be bool [32] with at least one valid token")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a nonempty string")
        self.target_id = target_id
        self.metadata = dict(metadata or {})
        self.register_buffer("padding_mask", mask.detach().clone())
        self.register_buffer("template", features.detach().float().clone())
        self.register_buffer("valid_indices", (~mask).nonzero().flatten())
        self.features = nn.Parameter(self.template[~mask].clone())

    def full_features(self) -> torch.Tensor:
        return self.template.index_copy(0, self.valid_indices, self.features)

    def forward(self, batch_size: int, device=None) -> dict:
        if batch_size < 1:
            raise ValueError("A learned prompt needs at least one query")
        features = self.full_features().to(device=device)
        mask = self.padding_mask.to(device=device)
        return {
            "language_features": features[:, None].expand(-1, batch_size, -1),
            "language_mask": mask[None].expand(batch_size, -1),
        }

    def require_target(self, target_id: str) -> None:
        if str(target_id) != self.target_id:
            raise ValueError(
                f"Loaded target is {self.target_id!r}, requested {target_id!r}; "
                "load that target's feature file in a separate model."
            )

    def save(self, path: str | Path, training_state: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "sam3_learned_prompt_v1",
            "target_id": self.target_id,
            "features": self.full_features().detach().cpu(),
            "mask": self.padding_mask.detach().cpu(),
            "metadata": self.metadata,
        }
        if training_state is not None:
            payload["training_state"] = training_state
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, checkpoint_path=None) -> LearnedPrompt:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "sam3_learned_prompt_v1":
            raise ValueError("Not a SAM3 learned prompt file")
        prompt = cls(
            payload["features"],
            payload["mask"],
            payload["target_id"],
            payload["metadata"],
        )
        if checkpoint_path is not None:
            expected = prompt.metadata.get("base_checkpoint_sha256")
            if not expected or checkpoint_sha256(checkpoint_path) != expected:
                raise ValueError("Learned prompt and base SAM3 checkpoint do not match")
        return prompt


def attach_learned_prompt(
    model: nn.Module, prompt: LearnedPrompt, training: bool = False
) -> nn.Module:
    """Called after loading the frozen base checkpoint, before moving the model."""
    backbone = model.detector.backbone if hasattr(model, "detector") else model.backbone
    if backbone.language_backbone is not None:
        raise ValueError("Build the learned mode without a text encoder")
    model.requires_grad_(False)
    backbone.learned_prompt = prompt
    prompt.requires_grad_(training)
    return model
