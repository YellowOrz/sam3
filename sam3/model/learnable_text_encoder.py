# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Dict, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn


class LearnableClassTextEncoder(nn.Module):
    """Replace natural-language encoding with learned hand-class tokens."""

    CLASS_NAMES = ("left_hand", "right_hand")
    PLACEHOLDER_NAMES = ("<text_placeholder>", "visual")

    def __init__(
        self,
        tokens_per_class: int = 1,
        d_model: int = 256,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if tokens_per_class < 1:
            raise ValueError("tokens_per_class must be at least 1")

        self.tokens_per_class = tokens_per_class
        self.d_model = d_model
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(self.CLASS_NAMES)
        }
        self.class_tokens = nn.Parameter(
            torch.empty(len(self.CLASS_NAMES), tokens_per_class, d_model)
        )
        nn.init.normal_(self.class_tokens, std=init_std)

    def forward(
        self,
        captions: Sequence[str],
        input_boxes: Optional[Sequence] = None,
        device: Union[str, torch.device, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_boxes is not None and len(input_boxes) > 0:
            raise ValueError("Learnable class tokens do not support input_boxes")
        if len(captions) == 0:
            raise ValueError("captions must contain at least one class name")

        normalized_captions = [self._normalize_class_name(name) for name in captions]
        unknown = [
            name
            for name in normalized_captions
            if name not in self.class_to_index and name not in self.PLACEHOLDER_NAMES
        ]
        if unknown:
            allowed = ", ".join(self.CLASS_NAMES)
            raise ValueError(f"Unknown class {unknown[0]!r}; expected one of: {allowed}")

        output_device = (
            self.class_tokens.device if device is None else torch.device(device)
        )
        feature_rows = [
            (
                torch.zeros(
                    self.tokens_per_class,
                    self.d_model,
                    device=self.class_tokens.device,
                )
                if name in self.PLACEHOLDER_NAMES
                else self.class_tokens[self.class_to_index[name]]
            )
            for name in normalized_captions
        ]
        features = torch.stack(feature_rows, dim=0).transpose(0, 1).to(output_device)
        padding_mask = torch.zeros(
            len(captions),
            self.tokens_per_class,
            dtype=torch.bool,
            device=output_device,
        )
        return padding_mask, features, features

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        return class_name.strip().lower().replace(" ", "_")


def freeze_for_learnable_class_tokens(model: nn.Module) -> Set[str]:
    """Freeze a model except for parameters owned by class-token encoders."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    found_encoder = False
    for module in model.modules():
        if isinstance(module, LearnableClassTextEncoder):
            found_encoder = True
            module.class_tokens.requires_grad = True

    if not found_encoder:
        raise ValueError("model does not contain a LearnableClassTextEncoder")

    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def filter_checkpoint_for_learnable_class_tokens(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Drop legacy text-encoder weights while retaining trained class tokens."""
    return {
        key: value
        for key, value in state_dict.items()
        if "language_backbone." not in key
        or key.endswith("language_backbone.class_tokens")
    }
