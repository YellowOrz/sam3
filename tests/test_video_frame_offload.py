import pytest
import torch

from sam3.model.data_misc import NestedTensor, gather_frames
from sam3.model.sam3_video_inference import Sam3VideoInference


def test_gather_frames_indexes_cpu_batch_with_cuda_ids() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    frames = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, 3, 2, 2)
    ids = torch.tensor([1, 3], device="cuda")
    got = gather_frames(frames, ids)
    assert got.device.type == "cpu"
    assert torch.equal(got, frames[[1, 3]])
    nested = NestedTensor(frames, mask=None)
    assert torch.equal(gather_frames(nested, ids), frames[[1, 3]])


def test_sam3_input_batch_does_not_copy_cpu_frames_to_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    images = torch.zeros(3, 3, 8, 8)
    state = {"constants": {}}
    host = type("Host", (), {"device": torch.device("cuda")})()

    Sam3VideoInference._construct_initial_input_batch(host, state, images)

    assert state["input_batch"].img_batch is images
    assert state["input_batch"].find_inputs[0].input_boxes.device.type == "cuda"
