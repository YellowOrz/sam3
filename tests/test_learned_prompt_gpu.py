"""Opt-in end-to-end test: SAM3_TEST_CHECKPOINT=/path/to/sam3.pt pytest ..."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not os.environ.get("SAM3_TEST_CHECKPOINT"),
    reason="Requires an explicit SAM3 checkpoint, CUDA and SAM3 training dependencies",
)
def test_real_sam3_positive_negative_training_and_resume(tmp_path, monkeypatch):
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from sam3.train import learned_prompt as training

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real SAM3 smoke test")
    checkpoint = os.environ["SAM3_TEST_CHECKPOINT"]
    spec = importlib.util.spec_from_file_location(
        "prepare_learned_prompt", ROOT / "scripts/prepare_learned_prompt.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    feature_file = tmp_path / "initial.pt"
    prepare.main(
        [
            "--checkpoint",
            checkpoint,
            "--target-id",
            "target",
            "--reference-text",
            "object",
            "--output",
            str(feature_file),
        ]
    )
    images = tmp_path / "images"
    images.mkdir()
    for index in (1, 2):
        Image.new("RGB", (64, 64), (64 * index, 80, 120)).save(images / f"{index}.png")
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": f"{i}.png", "height": 64, "width": 64}
                    for i in (1, 2)
                ],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [16, 16, 32, 32],
                        "area": 1024,
                        "iscrowd": 0,
                        "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                    }
                ],
            }
        )
    )
    config = OmegaConf.load(ROOT / "sam3/train/configs/learned_prompt.yaml")
    config.checkpoint = checkpoint
    config.initial_feature = str(feature_file)
    config.target_id = "target"
    config.output_dir = str(tmp_path / "run")
    config.epochs = 1
    config.train = {"images": str(images), "annotations": str(annotations)}
    config.val = config.train  # synthetic mechanics check, not a quality evaluation

    real_builder = training.build_sam3_image_model
    built = []

    def checked_builder(**kwargs):
        model = real_builder(**kwargs)
        assert model.backbone.language_backbone is None
        assert len([p for p in model.parameters() if p.requires_grad]) == 1
        built.append(model)
        return model

    monkeypatch.setattr(training, "build_sam3_image_model", checked_builder)
    training.train(config)
    model = built.pop()
    assert model.backbone.learned_prompt.features.grad is not None
    assert all(
        p.grad is None
        for n, p in model.named_parameters()
        if n != "backbone.learned_prompt.features"
    )
    initial = torch.load(feature_file, weights_only=True)["features"]
    result = torch.load(tmp_path / "run/learned_prompt.pt", weights_only=True)
    assert not torch.equal(initial, result["features"])
    assert torch.equal(initial[result["mask"]], result["features"][result["mask"]])
    del model
    torch.cuda.empty_cache()
    config.epochs = 2
    training.train(config, resume=str(tmp_path / "run/last.pt"))
    saved = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=True)
    assert saved["training_state"]["epoch"] == 2
