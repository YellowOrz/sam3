"""Evaluation-only adapters for the unmodified SAM3 video predictor."""
import numpy as np
import torch
from torch import nn
from scripts.cached_ve_text_features import CLASS_NAMES, NATURAL_PROMPTS


class VideoResidualTextEncoder(nn.Module):
    """Use trained hand cache while preserving actual VE for 'visual'/other text.

    The video API requests a visual placeholder alongside hand text. Silently
    mapping that placeholder to a hand would change the system baseline.
    """
    def __init__(self, original, cache):
        super().__init__()
        self.original = original
        self.cache = cache
        self.eval().requires_grad_(False)

    def forward(self, captions, input_boxes=None, device=None):
        padding, features, raw = self.original(captions, input_boxes=input_boxes, device=device)
        padding, features, raw = padding.clone(), features.clone(), raw.clone()
        aliases = {name: key for name, key in zip(NATURAL_PROMPTS, CLASS_NAMES)}
        aliases.update({key: key for key in CLASS_NAMES})
        for index, caption in enumerate(captions):
            if caption not in aliases:
                continue
            p, f, r = self.cache([aliases[caption]], device=device)
            if padding.shape[1] != p.shape[1] or features.shape[0] != f.shape[0] or raw.shape[0] != r.shape[0]:
                raise ValueError('Video VE/cache context length mismatch')
            padding[index] = p[0]
            features[:, index] = f[:, 0]
            raw[:, index] = r[:, 0]
        return padding, features, raw


def union_video_outputs(outputs, shape):
    """Use predictor's actual accepted masks; no GT matching or top-score pick."""
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    ids = array(outputs.get('out_obj_ids', [])).reshape(-1)
    scores = array(outputs.get('out_probs', [])).reshape(-1)
    masks = array(outputs.get('out_binary_masks', np.zeros((0, *shape), bool)))
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.shape != (len(ids), *shape) or len(scores) != len(ids):
        raise ValueError('Video frame output dimensions/counts mismatch')
    if not np.isfinite(scores).all() or not np.isfinite(masks).all():
        raise ValueError('Nonfinite video outputs')
    if len(set(ids.tolist())) != len(ids) or not np.isin(masks, [0,1]).all():
        raise ValueError('Duplicate IDs or nonbinary output masks')
    nonempty = masks.reshape(len(ids), -1).any(1) if len(ids) else np.zeros(0, bool)
    union = masks.astype(bool).any(0) if len(ids) else np.zeros(shape, bool)
    return union, [int(x) for x in ids[nonempty]], [float(x) for x in scores[nonempty]]


def select_video_indices(images, recordings, frame_limit=None):
    if not recordings or len(set(recordings)) != len(recordings):
        raise ValueError('Explicit unique recording selection required')
    if frame_limit is not None and (type(frame_limit) is not int or frame_limit < 1):
        raise ValueError('Positive frame limit required')
    result = {}
    for name in recordings:
        indices = [i for i, r in enumerate(images) if r['recording_id'] == name]
        if not indices or [images[i]['source_frame_index'] for i in indices] != list(range(len(indices))):
            raise ValueError('Require a complete contiguous published recording starting at zero')
        result[name] = indices[:frame_limit]
    return result
