"""Exercise validation logging with real CPU tensors and TensorBoard events."""

import ast
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def test_validation_images_and_logging_guards(tmp_path):
    # Follow the existing AST test pattern to avoid importing the GPU model stack.
    path = Path(__file__).resolve().parents[1] / "sam3/train/learned_prompt.py"
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_log_val_images", "run_epoch")
    ]
    namespace = dict(
        torch=torch,
        sys=sys,
        tqdm=tqdm,
        DummyTqdmFile=DummyTqdmFile,
        _unwrap=lambda model: model,
        _rank0=lambda: True,
        _run_vision_as_constant=lambda backbone: None,
        copy_data_to_device=lambda batch, device: batch,
        CORE_LOSS_KEY="loss",
        dist=SimpleNamespace(is_available=lambda: False),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + functions,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    image = torch.zeros(2, 3, 4, 4)
    image[0] = -1  # black; second image becomes gray after denormalization
    gt = torch.zeros(2, 4, 4, dtype=torch.bool)
    gt[0, 0, 0] = True
    gt[1, 3, 3] = True
    target = {"num_boxes": torch.tensor([0, 2]), "masks": gt}
    batch = SimpleNamespace(
        img_batch=image,
        find_targets=[target],
        find_inputs=[SimpleNamespace(img_ids=torch.tensor([1, 0]))],
    )
    output = {
        "pred_logits": torch.tensor([[[10.0], [10.0]], [[10.0], [-10.0]]]),
        "presence_logit_dec": torch.tensor([[-10.0], [10.0]]),
        "pred_masks": torch.full((2, 2, 2, 2), 10.0),
    }
    output["pred_masks"][1, 0, 1] = -10.0
    model = Mock()
    model.modules.return_value = []
    model.back_convert.side_effect = lambda value: value
    model.return_value = [output]
    loss_fn = lambda outputs, targets: {"loss": torch.tensor(2.0)}
    config = SimpleNamespace(device="cpu", amp=False, val_visualization_max_images=3)
    config.get = lambda key, default=None: getattr(config, key, default)
    run_epoch = namespace["run_epoch"]
    loader = [{"target": batch}, {"target": batch}]
    with SummaryWriter(str(tmp_path)) as writer:
        metrics, step = run_epoch(
            model,
            loader,
            loss_fn,
            config,
            tb_logger=SimpleNamespace(writer=writer),
            tb_step=12,
        )
    assert metrics == {"loss": 2.0} and step == 12
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Tags()["images"] == [f"val/images/{i:03d}" for i in range(3)]
    panels = []
    for i in range(3):
        (event,) = events.Images(f"val/images/{i:03d}")
        assert event.step == 12 and (event.height, event.width) == (4, 12)
        panels.append(np.array(Image.open(io.BytesIO(event.encoded_image_string))))
    # Query-to-image mapping, empty GT and presence suppression.
    assert np.all(panels[0] == 127)
    assert np.array_equal(panels[0], panels[2])
    assert not panels[1][:, :4].any()
    assert panels[1][0, 4].tolist() == [0, 127, 0]
    assert panels[1][3, 7].tolist() == [0, 127, 0]  # second GT instance
    assert panels[1][0, 8].tolist() == [0, 127, 0]
    assert not panels[1][3, 8:].any()  # low-score all-positive mask excluded
    config.val_visualization_threshold = 1.0
    config.val_visualization_max_images = 8
    writer = Mock()
    run_epoch(
        model,
        loader[:1],
        loss_fn,
        config,
        tb_logger=SimpleNamespace(writer=writer),
    )
    assert writer.add_image.call_count == 2
    for call in writer.add_image.call_args_list:
        panel = call.args[1]
        assert torch.equal(panel[:, :, :4], panel[:, :, 8:])
    # No work on non-primary ranks, disabled logging, or absent writer/logger.
    log_images = namespace["_log_val_images"] = Mock()
    for rank0, limit, logger in (
        (False, 3, SimpleNamespace(writer=Mock())),
        (True, 0, SimpleNamespace(writer=Mock())),
        (True, 3, SimpleNamespace(writer=None)),
        (True, 3, None),
    ):
        namespace["_rank0"] = lambda: rank0
        config.val_visualization_max_images = limit
        run_epoch(model, loader, loss_fn, config, tb_logger=logger)
    log_images.assert_not_called()
