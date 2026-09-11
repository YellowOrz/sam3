"""Experimental MANO-parameter tokens for explicit, opt-in prompt injection.

The output layout matches ``SequenceGeometryEncoder``: sequence-first features
and a batch-first PyTorch padding mask (True means ignore). This module consumes
one already decoded MANO hand per *prompt*, not per distinct image. A caller with
two side queries for one image must explicitly align/duplicate the appropriate
records; the side here describes the supplied hand, not the queried class.

No MANO layer, PCA decoding, joint projection, camera transform, image-feature
cross-attention, or spatial alignment check is performed here. In particular,
do not pass the raw dataset ``pose_m`` array as ``hand_pose`` without first
establishing its representation and converting it to the contract below.

``mano_prompt_adapter`` supplies an optional geometry-output wrapper. The
default builder and ordinary token training do not install it or load MANO.
"""

import math
from typing import Optional, Set, Tuple

import torch
from torch import nn


class ManoGeometryEncoder(nn.Module):
    """Encode normalized MANO fields as five learnable geometric tokens.

    Input tensors are batch-first and share the module's device:

    * global_orient: float [B, 3], MANO root axis-angle in radians in the camera
      coordinate frame.
    * hand_pose: float [B, 45], the 15 MANO local joint axis-angle rotations in
      radians, in MANO joint order, with any PCA/pose-mean convention resolved.
    * betas: float [B, 10], MANO shape coefficients for the stated side/model.
    * transl: float [B, 3], MANO translation in metres in the camera coordinate
      frame. This is a translation feature, not a projected wrist/joint token.
    * side: int64 [B], LEFT=0, RIGHT=1; MISSING=-1 is allowed only for invalid
      rows. This is physical hand side, not screen position or calibration name.
    * valid: optional bool [B]; False means *all* MANO input for this row is
      missing/unusable. It is not mask visibility: an occluded hand may still
      have valid MANO parameters. Omission means every row is valid.

    Valid float rows must be finite. Invalid rows may contain NaN placeholders;
    they are zeroed before encoding and receive zero feature/input gradients.
    Floating inputs are cast to the module dtype without detaching gradients.

    Returns ``(features, padding_mask)`` with shapes [5, B, d_model] and [B, 5].
    Token order is ``TOKEN_FIELDS``; d_model defaults to SAM3's 256 channels.
    All tokens of an invalid row are masked. The integrating caller must retain
    an unmasked text/null token or skip attention for such a row: feeding only an
    all-masked geometry sequence to attention can produce undefined outputs.

    This class has the geometry encoder's *output* convention, not its ``Prompt``
    input signature. Use the explicit wrapper in ``mano_prompt_adapter`` with
    correctly aligned ``ManoPrompt`` inputs; this class alone is not a drop-in
    replacement and does not install data/collator/model wiring.
    """

    LEFT = 0
    RIGHT = 1
    MISSING = -1
    TOKEN_FIELDS = ("global_orient", "hand_pose", "betas", "transl", "side")
    FIELD_DIMS = {"global_orient": 3, "hand_pose": 45, "betas": 10, "transl": 3}

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        translation_scale_m: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not math.isfinite(translation_scale_m) or translation_scale_m <= 0:
            raise ValueError("translation_scale_m must be finite and positive")
        self.d_model = d_model
        self.tokens_per_prompt = len(self.TOKEN_FIELDS)
        self.field_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for name, input_dim in self.FIELD_DIMS.items()
            }
        )
        self.side_embedding = nn.Embedding(2, d_model)
        self.field_embedding = nn.Parameter(torch.empty(self.tokens_per_prompt, d_model))
        nn.init.normal_(self.field_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(d_model)
        # Keep unit normalization in state_dict so checkpoint loading preserves it.
        self.register_buffer("translation_scale_m", torch.tensor(float(translation_scale_m)))

    def forward(
        self,
        *,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        side: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.field_embedding.device
        if not isinstance(side, torch.Tensor) or side.dtype != torch.long or side.ndim != 1:
            raise ValueError("side must be an int64 tensor with shape [B]")
        batch_size = side.shape[0]
        if batch_size < 1:
            raise ValueError("MANO prompt batch must contain at least one row")
        if side.device != device:
            raise ValueError("side and encoder must be on the same device")
        if valid is None:
            valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != (batch_size,)
            or valid.device != device
        ):
            raise ValueError("valid must be a bool tensor with shape [B] on the encoder device")
        known_side = (side == self.LEFT) | (side == self.RIGHT)
        if not bool((known_side | ((side == self.MISSING) & ~valid)).all()):
            raise ValueError("side must be LEFT=0 or RIGHT=1; -1 is allowed only when valid=False")

        fields = {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas,
            "transl": transl,
        }
        encoded = []
        for name, values in fields.items():
            expected_shape = (batch_size, self.FIELD_DIMS[name])
            if (
                not isinstance(values, torch.Tensor)
                or values.shape != expected_shape
                or not values.is_floating_point()
                or values.device != device
            ):
                raise ValueError(
                    f"{name} must be a floating tensor with shape {expected_shape} "
                    "on the encoder device"
                )
            if not bool(torch.isfinite(values[valid]).all()):
                raise ValueError(f"{name} contains nonfinite values in valid MANO rows")
            # Mask before any linear operation: NaN * 0 can otherwise poison
            # parameter gradients even when the final output is masked out.
            values = torch.where(valid[:, None], values, torch.zeros_like(values))
            values = values.to(dtype=self.field_embedding.dtype)
            if name == "transl":
                values = values / self.translation_scale_m
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{name} overflows the encoder dtype or normalization")
            encoded.append(self.field_encoders[name](values))

        safe_side = torch.where(valid, side, torch.zeros_like(side))
        encoded.append(self.side_embedding(safe_side))
        features = torch.stack(encoded, dim=0) + self.field_embedding[:, None, :]
        features = self.output_norm(features)
        features = features.masked_fill(~valid[None, :, None], 0.0)
        padding_mask = (~valid[:, None]).expand(batch_size, self.tokens_per_prompt)
        return features, padding_mask


def freeze_for_mano_geometry_encoder(model: nn.Module) -> Set[str]:
    """Freeze all weights except registered ``ManoGeometryEncoder`` modules.

    This is for the MANO-only adaptation stage; existing class tokens are frozen
    too. It does not set eval/train modes, construct an optimizer, clear existing
    gradients, or install an encoder. Call before constructing a fresh optimizer.
    Frozen downstream modules must still run with autograd enabled so that loss
    gradients can propagate back to the adapter.
    """
    encoders = [module for module in model.modules() if isinstance(module, ManoGeometryEncoder)]
    if not encoders:
        raise ValueError("model does not contain a ManoGeometryEncoder")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for encoder in encoders:
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}
