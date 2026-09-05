"""CPU tests for real tensor assembly, source snapshots and the opt-in hook."""

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "memory_under_test", ROOT / "sam3/model/bidirectional_memory.py"
)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class Tracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.mem_dim = 2
        self.num_maskmem = 4
        self.max_obj_ptrs_in_encoder = 4
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(4, 1, 1, 2))
        self.encoder_calls = []
        self.decoder_calls = []
        self.transformer = SimpleNamespace(encoder=self.encoder)
        self.eval()

    def encoder(self, **kwargs):
        self.encoder_calls.append(kwargs)
        # Both banks affect a newly computed feature, not a selected source mask.
        value = kwargs["prompt"][: -kwargs["num_obj_ptr_tokens"]].mean()
        return {"memory": torch.ones(4, 1, 4) * value}

    def _get_tpos_enc(self, distances, **kwargs):
        assert all(0 < d <= 3 for d in distances)
        return torch.zeros(len(distances), 2)

    def _use_multimask(self, *args):
        return False

    def _forward_sam_heads(self, **kwargs):
        self.decoder_calls.append(kwargs)
        assert not torch.is_grad_enabled()
        value = kwargs["backbone_features"].mean().item()
        low = torch.full((1, 1, 2, 2), value)
        return None, None, torch.tensor([[0.8]]), low, low, None, torch.tensor([[2.0]])


def write_entry(tmp_path, frame, direction, value, pointer):
    path = tmp_path / f"{direction}_{frame}.pt"
    torch.save(
        {
            "maskmem_features": torch.full((1, 2, 2, 2), value),
            "maskmem_pos_enc": [torch.zeros(1, 2, 2, 2)],
            "obj_ptr": torch.full((1, 4), pointer),
        },
        path,
    )
    return {
        "path": str(path),
        "frame_index": frame,
        "direction": direction,
        "conditioning": False,
        "spatial_tokens": 4,
    }


def test_joint_attention_has_pointer_tail_one_decoder_and_frozen_sources(tmp_path):
    tracker = Tracker()
    first = write_entry(tmp_path, 0, "F", 2.0, 100.0)
    second = write_entry(tmp_path, 9, "B", 6.0, 200.0)
    files_before = {
        e["path"]: hashlib.sha256(Path(e["path"]).read_bytes()).digest()
        for e in (first, second)
    }
    weights_before = {
        name: tensor.clone() for name, tensor in tracker.state_dict().items()
    }
    result = memory.read_memory(
        tracker,
        4,
        [torch.zeros(4, 1, 4)],
        [torch.zeros(4, 1, 4)],
        [(2, 2)],
        [first, second],
        [first, second],
    )
    assert len(tracker.encoder_calls) == len(tracker.decoder_calls) == 1
    call = tracker.encoder_calls[0]
    assert call["num_obj_ptr_tokens"] == 4
    assert torch.all(call["prompt"][:4] == 2)
    assert torch.all(call["prompt"][4:8] == 6)
    assert torch.all(call["prompt"][8:10] == 100)
    assert torch.all(call["prompt"][10:] == 200)
    assert torch.all(result["logits"] == 4)
    assert result["spatial_tokens"] == 8 and result["pointer_tokens"] == 4
    assert all(
        torch.equal(weights_before[n], v) for n, v in tracker.state_dict().items()
    )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).digest() == digest
        for p, digest in files_before.items()
    )
    with pytest.raises(ValueError, match="own"):
        memory.read_memory(
            tracker,
            0,
            [torch.zeros(4, 1, 4)],
            [torch.zeros(4, 1, 4)],
            [(2, 2)],
            [first],
            [],
        )
    with pytest.raises(ValueError, match="direction"):
        memory.read_memory(
            tracker,
            4,
            [torch.zeros(4, 1, 4)],
            [torch.zeros(4, 1, 4)],
            [(2, 2)],
            [{**first, "direction": "B"}],
            [],
        )


def test_export_slices_batched_objects_and_survives_source_mutation(tmp_path):
    output = {
        "maskmem_features": torch.arange(16.0).reshape(2, 2, 2, 2),
        "maskmem_pos_enc": [torch.zeros(2, 2, 2, 2)],
        "obj_ptr": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        "iou_score": torch.tensor([0.8, 0.3]),
        "object_score_logits": torch.tensor([[3.0], [-4.0]]),
    }
    state = {
        "memory_export_directory": str(tmp_path),
        "memory_export_records": {},
        "tracker_inference_states": [
            {
                "obj_id_to_idx": {7: 0, 99: 1},
                "output_dict": {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {2: output},
                },
            }
        ],
    }
    memory.export_memory_frame(state, 2)
    records = state["memory_export_records"][2]
    assert records[99]["quality"] == pytest.approx(0.3)
    assert records[99]["presence_logit"] == -4
    output["maskmem_features"].zero_()
    payload = torch.load(records[99]["path"], weights_only=True)
    assert torch.equal(payload["obj_ptr"], torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
    assert payload["maskmem_features"].shape == (1, 2, 2, 2)
    assert payload["maskmem_features"].sum() > 0


def test_normal_sam3_frame_does_not_enter_memory_capture(monkeypatch):
    # Compile the actual orchestration method without loading a GPU model or its
    # optional training dependencies. Exercise both branches on the same stub.
    tree = ast.parse(
        (ROOT / "sam3/model/sam3_video_inference.py").read_text(encoding="utf-8")
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Sam3VideoInference"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_single_frame_inference"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "actual_frame_method", "exec"
        ),
        namespace,
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "sam3.model.bidirectional_memory",
        SimpleNamespace(export_memory_frame=lambda state, index: calls.append(index)),
    )
    model = SimpleNamespace(rank=1)
    model._det_track_one_frame = lambda **kwargs: (
        {},
        {},
        [],
        {"obj_id_to_tracker_score_frame_wise": {0: {}}},
        {},
        None,
    )
    state = {
        "input_batch": None,
        "tracker_inference_states": [],
        "text_prompt": None,
        "per_frame_geometric_prompt": [None],
        "constants": {"empty_geometric_prompt": None},
        "num_frames": 1,
        "tracker_metadata": {},
        "feature_cache": {},
        "orig_height": 2,
        "orig_width": 2,
        "is_image_only": False,
        "previous_stages_out": [None],
    }
    baseline = namespace[method.name](model, state, 0, False)
    assert calls == []
    state["memory_export_directory"] = "enabled"
    captured = namespace[method.name](model, state, 0, False)
    assert calls == [0]
    assert baseline == captured
