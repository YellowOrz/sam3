"""Opt-in MANO prompt transport and geometry-output wrapper.

Default builders, data loaders and ordinary token training do not install this
experimental path. Load base weights first, explicitly wrap ``geometry_encoder``
with ``ManoAugmentedGeometryEncoder``, and supply a per-query ``ManoPrompt`` to
``forward_grounding``. The separate MANO engineering smoke script demonstrates
this opt-in path; its existence does not make ordinary training consume a sidecar
or establish a completed MANO fine-tuning experiment.
"""

from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn

from .geometry_encoders import Prompt, concat_padded_sequences
from .mano_geometry_encoder import ManoGeometryEncoder


class ManoPrompt(Prompt):
    """Keep ordinary geometric prompts and normalized per-query MANO together.

    ``mano`` is a dict accepted by ``ManoGeometryEncoder.forward``. Its tensors
    must already follow that encoder's units, representation, device and prompt
    ordering. This class does not infer which MANO record belongs to which side
    query. ``clone`` preserves all point/box/mask fields and MANO gradients.

    A MANO-only constructor creates empty point prompts to establish the batch
    shape expected by the existing geometry encoder. For existing prompts use
    ``ManoPrompt.from_prompt(prompt, mano=...)``; that method clones tensor storage.
    """

    PROMPT_FIELDS = (
        "box_embeddings", "box_mask", "box_labels",
        "point_embeddings", "point_mask", "point_labels",
        "mask_embeddings", "mask_mask", "mask_labels",
    )
    REQUIRED_MANO_FIELDS = frozenset({"global_orient", "hand_pose", "betas", "transl", "side"})

    def __init__(self, *, mano: Optional[Mapping] = None, **prompt_fields):
        if mano is not None:
            if not isinstance(mano, Mapping):
                raise ValueError("mano must be a tensor mapping or None")
            missing = self.REQUIRED_MANO_FIELDS - mano.keys()
            unknown = mano.keys() - self.REQUIRED_MANO_FIELDS - {"valid"}
            if missing or unknown:
                raise ValueError(f"Invalid MANO fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
            for name, value in mano.items():
                if not isinstance(value, torch.Tensor) and not (name == "valid" and value is None):
                    raise ValueError(f"mano[{name!r}] must be a tensor")
            side = mano["side"]
            if side.ndim != 1 or side.dtype != torch.long or side.numel() < 1:
                raise ValueError("mano side must be a nonempty int64 tensor with shape [B]")
            if all(prompt_fields.get(name) is None for name in (
                "box_embeddings", "point_embeddings", "mask_embeddings"
            )):
                prompt_fields["point_embeddings"] = torch.empty(
                    0, side.shape[0], 2,
                    dtype=mano["global_orient"].dtype,
                    device=side.device,
                )
        super().__init__(**prompt_fields)
        self.mano = None if mano is None else dict(mano)

    @staticmethod
    def _clone_tensor(value):
        return None if value is None else value.clone()

    @classmethod
    def from_prompt(cls, prompt: Prompt, *, mano: Optional[Mapping] = None):
        """Clone all geometry fields and attach cloned normalized MANO tensors."""
        fields = {name: cls._clone_tensor(getattr(prompt, name)) for name in cls.PROMPT_FIELDS}
        copied_mano = None if mano is None else {
            name: cls._clone_tensor(value) for name, value in mano.items()
        }
        return cls(mano=copied_mano, **fields)

    def clone(self):
        return self.from_prompt(self, mano=self.mano)


class ManoAugmentedGeometryEncoder(nn.Module):
    """Append MANO tokens to the unchanged base geometry encoder's outputs.

    A plain Prompt or ``ManoPrompt(mano=None)`` follows the exact base forward
    path. With MANO, the original geometric tokens remain and five new tokens
    per query are appended while keeping padding on the right. This is a token
    injection prototype, not a completed replacement with projected 2D joints
    or MANO/image spatial cross-attention inside the geometry encoder.

    Wrapping changes checkpoint names from ``geometry_encoder.<name>`` to
    ``geometry_encoder.base_encoder.<name>``. Load the base checkpoint before
    wrapping; save/restore experiment weights with the same explicit wrapper.
    Use ``freeze_for_mano_geometry_encoder`` before creating the experiment's
    optimizer to freeze the base geometry encoder and the remaining model.
    """

    def __init__(self, base_encoder: nn.Module, mano_encoder: Optional[ManoGeometryEncoder] = None):
        super().__init__()
        self.base_encoder = base_encoder
        self.mano_encoder = ManoGeometryEncoder() if mano_encoder is None else mano_encoder
        if hasattr(base_encoder, "d_model") and base_encoder.d_model != self.mano_encoder.d_model:
            raise ValueError("Base geometry and MANO encoder d_model must match")

    @property
    def mask_encoder(self):
        """Keep SAM3Image's existing previous-mask downsampling access valid."""
        return self.base_encoder.mask_encoder

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        base_output = self.base_encoder(
            geo_prompt=geo_prompt,
            img_feats=img_feats,
            img_sizes=img_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        mano = getattr(geo_prompt, "mano", None)
        if mano is None:
            return base_output
        geo_features, geo_mask = base_output
        mano_features, mano_mask = self.mano_encoder(**mano)
        if geo_features.shape[1:] != mano_features.shape[1:]:
            raise ValueError("Geometry and MANO batch/channel shapes differ; align MANO per query")
        if geo_features.device != mano_features.device:
            raise ValueError("Geometry and MANO features must be on the same device")
        # Base geometric tokens may have autocast dtype even when MANO weights
        # remain FP32. Casting preserves the adapter's autograd path.
        mano_features = mano_features.to(dtype=geo_features.dtype)
        return concat_padded_sequences(geo_features, geo_mask, mano_features, mano_mask)


def attach_mano_geometry_encoder(
    model: nn.Module, mano_encoder: Optional[ManoGeometryEncoder] = None
) -> ManoAugmentedGeometryEncoder:
    """Explicitly wrap an already-loaded image model or video ``model.detector``.

    Call after loading base/model-token checkpoints and before constructing the
    experiment optimizer. This changes the registered geometry checkpoint key
    prefix; it does not load weights, freeze parameters, or populate ManoPrompt.
    A newly created adapter is moved to the base encoder's device/dtype. For an
    explicitly provided adapter, device and dtype placement remain caller-owned.
    """
    base = getattr(model, "geometry_encoder", None)
    if not isinstance(base, nn.Module):
        raise ValueError("model must expose a geometry_encoder module; pass the video detector")
    if isinstance(base, ManoAugmentedGeometryEncoder):
        raise ValueError("model already has a ManoAugmentedGeometryEncoder")
    if mano_encoder is None:
        mano_encoder = ManoGeometryEncoder(d_model=getattr(base, "d_model", 256))
        reference = next(base.parameters(), None)
        if reference is not None:
            mano_encoder.to(device=reference.device, dtype=reference.dtype)
    wrapper = ManoAugmentedGeometryEncoder(base, mano_encoder)
    wrapper.train(base.training)
    model.geometry_encoder = wrapper
    return wrapper
