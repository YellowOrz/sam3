"""Opt-in residuals on input word embeddings of the frozen original VE.

This is an input-embedding residual experiment, not standard CoOp: it adds no
context tokens and does not replace the transformer. Two word-position residuals
are shared across left/right, while original side-word embeddings remain distinct.
Other text prompts pass through the original VE without an embedding update.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from torch import nn

from sam3.model.text_encoder_ve import VETextEncoder


FORMAT = "sam3-shared-input-ve-residual-v1"
PROMPTS = ("left hand", "right hand")
ALIASES = {"left_hand": "left hand", "right_hand": "right hand"}
POSITIONS = (1, 2)


class SharedInputVETextEncoder(nn.Module):
    """Only input_delta[side-word/hand-word, width] is optionally trainable.

The two rows are *word roles*, never left/right classes. The underlying original
embedding table, positional embeddings, transformer, layer norm and resizer stay
frozen. Initial exact equivalence is tested under the same numerical mode; a
different inference/training attention kernel need not be bitwise identical.
"""

    def __init__(self, original: VETextEncoder):
        super().__init__()
        if not isinstance(original, VETextEncoder):
            raise TypeError("Require the original VETextEncoder, not a feature cache")
        if original.encoder.pool_type != "none" or not original.encoder.output_tokens:
            raise ValueError("This adapter requires the unpooled VE token contract")
        if original.context_length != 32:
            raise ValueError("The hand prompt protocol uses context_length=32")
        self.original_ve = original
        self.context_length = original.context_length
        self.width = original.encoder.width
        ids = original.tokenizer(list(PROMPTS), context_length=32)
        if (tuple(ids.shape) != (2, 32) or (ids != 0).sum(1).tolist() != [4, 4]
                or not torch.equal((ids != 0).nonzero()[:, 1].reshape(2, 4), torch.arange(4).expand(2, 4))
                or ids[0, 1] == ids[1, 1] or ids[0, 2] != ids[1, 2]):
            raise ValueError("Natural hand tokenization no longer has the verified side/hand positions")
        device = original.encoder.token_embedding.weight.device
        self.register_buffer("natural_token_ids", ids.to(device=device).clone())
        self.input_delta = nn.Parameter(torch.zeros(2, self.width, dtype=torch.float32, device=device))
        self.original_ve.requires_grad_(False)
        self.eval()

    def _validate_mode(self):
        if any(module.training for module in self.modules()):
            raise RuntimeError("Keep original VE and the residual adapter in eval mode")
        if any(parameter.requires_grad for parameter in self.original_ve.parameters()):
            raise RuntimeError("Original VE must remain completely frozen")
        if self.input_delta.dtype != torch.float32 or not bool(torch.isfinite(self.input_delta).all()):
            raise RuntimeError("Input residual must remain finite FP32")

    def forward(self, text, input_boxes=None, device=None):
        self._validate_mode()
        if not isinstance(text, (list, tuple)) or not text:
            raise ValueError("Expected a nonempty prompt sequence")
        if not isinstance(text[0], str):
            # Preserve original already-encoded feature inputs unchanged.
            return self.original_ve(text, input_boxes=input_boxes, device=device)
        if any(not isinstance(value, str) for value in text):
            raise ValueError("Text prompts cannot mix string and encoded inputs")
        if input_boxes is not None and len(input_boxes):
            raise ValueError("Original string VE does not accept geometry boxes")
        captions = [ALIASES.get(value, value) for value in text]
        hand_rows = [index for index, value in enumerate(captions) if value in PROMPTS]
        if not hand_rows:
            return self.original_ve(captions, input_boxes=input_boxes, device=device)
        target_device = self.input_delta.device if device is None else torch.device(device)
        if target_device != self.input_delta.device:
            # torch.device('cuda') and cuda:0 address the same current device.
            if not (target_device.type == "cuda" and target_device.index is None
                    and self.input_delta.device.index == torch.cuda.current_device()):
                raise ValueError("Move adapter/model explicitly to the requested device")
        ve, encoder = self.original_ve, self.original_ve.encoder
        ids = ve.tokenizer(captions, context_length=self.context_length).to(self.input_delta.device)
        for index in hand_rows:
            expected = self.natural_token_ids[PROMPTS.index(captions[index])]
            if not torch.equal(ids[index], expected):
                raise RuntimeError("Tokenizer changed after initialization")
        embeds = encoder.token_embedding(ids)
        if embeds.dtype != torch.float32:
            raise RuntimeError("Keep original embedding weights FP32; use BF16 autocast, not model.bfloat16()")
        # Scatter only into exact hand query rows; object prompts get zero change.
        row_mask = torch.zeros(len(captions), device=embeds.device, dtype=embeds.dtype)
        row_mask[hand_rows] = 1
        position_basis = torch.zeros(2, ids.shape[1], device=embeds.device, dtype=embeds.dtype)
        position_basis[torch.arange(2, device=embeds.device), list(POSITIONS)] = 1
        # Disable autocast here to retain original FP32 input embeddings.
        with torch.autocast(device_type=embeds.device.type, enabled=False):
            update = position_basis.transpose(0, 1) @ self.input_delta
            embeds = embeds + row_mask[:, None, None] * update[None]
        attention = encoder.attn_mask
        if attention is not None:
            attention = attention[:ids.shape[1], :ids.shape[1]]
        hidden = embeds + encoder.positional_embedding[:ids.shape[1]]
        hidden = encoder.transformer(hidden, attn_mask=attention)
        hidden = encoder.ln_final(hidden)
        # Original pool_type=none returns this same tensor as output_tokens;
        # its separately computed pooled projection is unused by VE.forward.
        features = ve.resizer(hidden.transpose(0, 1))
        return ids.eq(0), features, embeds.transpose(0, 1)

    def residual_state(self):
        """Small checkpoint; NEVER serialize the frozen original VE weights."""
        return {"format": FORMAT, "input_delta": self.input_delta.detach().cpu().clone(),
                "natural_token_ids": self.natural_token_ids.detach().cpu().clone(),
                "width": self.width, "positions": list(POSITIONS), "prompts": list(PROMPTS),
                "shared_across_sides": True, "context_length": self.context_length}

    def load_residual_state(self, state):
        if (state.get("format") != FORMAT or state.get("width") != self.width
                or state.get("positions") != list(POSITIONS) or state.get("prompts") != list(PROMPTS)
                or state.get("shared_across_sides") is not True or state.get("context_length") != 32):
            raise ValueError("Input residual architecture metadata differs")
        delta, ids = state.get("input_delta"), state.get("natural_token_ids")
        if (not isinstance(delta, torch.Tensor) or delta.shape != self.input_delta.shape
                or delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all())
                or not isinstance(ids, torch.Tensor) or ids.dtype != self.natural_token_ids.dtype
                or not torch.equal(ids.cpu(), self.natural_token_ids.cpu())):
            raise ValueError("Invalid input residual tensor or tokenizer identity")
        with torch.no_grad():
            self.input_delta.copy_(delta.to(self.input_delta.device))


def install_shared_input_ve(model):
    original = model.backbone.language_backbone
    adapter = SharedInputVETextEncoder(original)
    model.backbone.language_backbone = adapter
    set_input_ve_training_mode(model, train_residual=False)
    return adapter


def set_input_ve_training_mode(model, *, train_residual):
    adapter = model.backbone.language_backbone
    if not isinstance(adapter, SharedInputVETextEncoder):
        raise TypeError("Install SharedInputVETextEncoder first")
    model.eval()
    model.requires_grad_(False)
    adapter.input_delta.requires_grad_(train_residual)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {"backbone.language_backbone.input_delta"} if train_residual else set()
    if names != expected:
        raise RuntimeError("Unexpected trainable parameter names")
    return names
