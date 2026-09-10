"""Small real-tensor tests; no SAM3 weights, CUDA, or eager sam3 import needed."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "learned_prompt_under_test", ROOT / "sam3/model/learned_prompt.py"
)
learned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learned)


def make_prompt(target="left", metadata=None):
    mask = torch.arange(32) >= 4
    return learned.LearnedPrompt(torch.randn(32, 256), mask, target, metadata)


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method_name]


def test_only_selected_feature_updates_through_frozen_decoder():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.language_backbone = None
    model.decoder = torch.nn.Linear(256, 1)
    prompt, other = make_prompt(), make_prompt("right")
    learned.attach_learned_prompt(model, prompt, training=True)
    original_weights = model.decoder.weight.detach().clone()
    original_feature = prompt.full_features().detach().clone()
    original_other = other.full_features().detach().clone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 1 and trainable[0] is prompt.features
    optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0.1)
    batch = prompt(3)
    assert batch["language_features"].shape == (32, 3, 256)
    assert batch["language_mask"].shape == (3, 32)
    loss = model.decoder(batch["language_features"][:4]).square().mean()
    loss.backward()
    assert prompt.features.grad.abs().sum() > 0
    assert model.decoder.weight.grad is None
    optimizer.step()
    assert torch.equal(model.decoder.weight, original_weights)
    assert not torch.equal(prompt.full_features()[:4], original_feature[:4])
    assert torch.equal(prompt.full_features()[4:], original_feature[4:])
    assert torch.equal(other.full_features(), original_other)


def test_text_initialization_and_batch_expansion_are_exact():
    prompt = make_prompt()
    features = prompt.full_features().detach()
    loaded = learned.LearnedPrompt(features, prompt.padding_mask, "left")
    assert torch.equal(loaded(2)["language_features"][:, 1], features)
    assert torch.equal(loaded(2)["language_mask"][1], prompt.padding_mask)


def test_feature_and_optimizer_resume(tmp_path):
    base = tmp_path / "base.pt"
    base.write_bytes(b"base checkpoint fingerprint")
    prompt = make_prompt(
        metadata={"base_checkpoint_sha256": learned.checkpoint_sha256(base)}
    )
    optimizer = torch.optim.AdamW([prompt.features], lr=0.01)
    prompt.features.square().sum().backward()
    optimizer.step()
    path = tmp_path / "last.pt"
    prompt.save(path, {"optimizer": optimizer.state_dict(), "epoch": 1})
    restored = learned.LearnedPrompt.load(path, base)
    next_optimizer = torch.optim.AdamW([restored.features], lr=0.01)
    state = torch.load(path, weights_only=True)["training_state"]
    next_optimizer.load_state_dict(state["optimizer"])
    for current, opt in ((prompt, optimizer), (restored, next_optimizer)):
        opt.zero_grad()
        current.features.square().sum().backward()
        opt.step()
    assert torch.equal(prompt.full_features(), restored.full_features())
    base.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="do not match"):
        learned.LearnedPrompt.load(path, base)


def test_fused_mlp_allows_global_grad_when_tensors_are_frozen():
    from sam3.perflib.fused import addmm_act

    linear = torch.nn.Linear(8, 8)
    linear.requires_grad_(False)
    inputs = torch.randn(2, 8)
    with torch.enable_grad():
        output = addmm_act(torch.nn.GELU, linear, inputs)
    assert output.shape == (2, 8)
    assert not output.requires_grad
    linear.weight.requires_grad_(True)
    with torch.enable_grad(), pytest.raises(ValueError, match="grad"):
        addmm_act(torch.nn.GELU, linear, inputs)


@pytest.mark.parametrize("bad", ["shape", "mask", "all_masked", "nan", "target"])
def test_invalid_features_rejected(bad):
    features, mask, target = (
        torch.zeros(32, 256),
        torch.zeros(32, dtype=torch.bool),
        "left",
    )
    if bad == "shape":
        features = features[:1]
    elif bad == "mask":
        mask = mask.float()
    elif bad == "all_masked":
        mask[:] = True
    elif bad == "nan":
        features[0, 0] = float("nan")
    else:
        target = ""
    with pytest.raises(ValueError):
        learned.LearnedPrompt(features, mask, target)


def test_default_backbone_still_calls_original_text_path():
    calls = []
    forward = load_method(
        "sam3/model/vl_combiner.py",
        "SAM3VLBackbone",
        "forward_text",
        {"activation_ckpt_wrapper": lambda fn: fn},
    )
    backbone = SimpleNamespace(
        training=False,
        act_ckpt_whole_language_backbone=False,
        _forward_text_no_ack_ckpt=lambda **kwargs: calls.append(kwargs) or "original",
    )
    assert forward(backbone, ["left hand"], device="cpu") == "original"
    assert calls[0]["captions"] == ["left hand"]
    backbone.learned_prompt = make_prompt()
    result = forward(backbone, ["left", "left"], device="cpu")
    assert len(calls) == 1
    assert result["language_features"].requires_grad
    with pytest.raises(ValueError):
        forward(backbone, ["left"], additional_text=["right"], device="cpu")


def test_image_text_compatibility_and_target_validation():
    calls = []
    namespace = {"Dict": dict}
    text = load_method(
        "sam3/model/sam3_image_processor.py",
        "Sam3Processor",
        "set_text_prompt",
        namespace,
    )
    feature = load_method(
        "sam3/model/sam3_image_processor.py",
        "Sam3Processor",
        "set_learned_prompt",
        namespace,
    )
    backbone = SimpleNamespace(forward_text=lambda *a, **kw: calls.append(a) or {})
    processor = SimpleNamespace(
        model=SimpleNamespace(backbone=backbone, _get_dummy_prompt=lambda: "geometry"),
        device="cpu",
        _forward_grounding=lambda state: state,
    )
    state = {"backbone_out": {}}
    text(processor, "left hand", state)
    assert calls == [(["left hand"],)]
    backbone.learned_prompt = make_prompt()
    with pytest.raises(ValueError, match="set_learned_prompt"):
        text(processor, "left hand", state)
    with pytest.raises(ValueError, match="Loaded target"):
        feature(processor, "right", state)
    feature(processor, "left", state)
    assert state["backbone_out"]["language_features"].shape == (32, 1, 256)


def test_builder_skips_encoder_only_when_opted_in(tmp_path, monkeypatch):
    tree = ast.parse((ROOT / "sam3/model_builder.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_sam3_image_model"
    )
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.decoder = torch.nn.Linear(256, 1)

    def backbone(vision, text):
        module = torch.nn.Module()
        module.language_backbone = text
        return module

    def text_encoder(path):
        calls.append("encoder")
        return torch.nn.Linear(1, 1)

    namespace = {
        "torch": torch,
        "_DEFAULT_BPE_PATH": "unused",
        "_create_vision_backbone": lambda **kw: None,
        "_create_text_encoder": text_encoder,
        "_create_vl_backbone": backbone,
        "_create_sam3_transformer": lambda: None,
        "_create_dot_product_scoring": lambda: None,
        "_create_segmentation_head": lambda **kw: None,
        "_create_geometry_encoder": lambda: None,
        "_create_sam3_model": lambda backbone, *args: Model(backbone),
        "_load_checkpoint": lambda model, path, skip_text_encoder: calls.append(
            skip_text_encoder
        ),
        "_setup_device_and_mode": lambda model, *args: model,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "builder", "exec"),
        namespace,
    )
    monkeypatch.setitem(sys.modules, "sam3.model.learned_prompt", learned)
    base = tmp_path / "base.pt"
    base.write_bytes(b"frozen checkpoint")
    feature = tmp_path / "feature.pt"
    make_prompt(
        metadata={"base_checkpoint_sha256": learned.checkpoint_sha256(base)}
    ).save(feature)
    build = namespace["build_sam3_image_model"]
    ordinary = build(checkpoint_path=base, device="cpu")
    assert ordinary.backbone.language_backbone is not None
    assert not hasattr(ordinary.backbone, "learned_prompt")
    assert calls == ["encoder", False]
    calls.clear()
    new = build(
        checkpoint_path=base, device="cpu", learned_prompt_path=feature, eval_mode=False
    )
    assert new.backbone.language_backbone is None
    assert calls == [True]
    assert [name for name, p in new.named_parameters() if p.requires_grad] == [
        "backbone.learned_prompt.features"
    ]


def test_video_request_compatibility_and_target_selection():
    tree = ast.parse(
        (ROOT / "sam3/model/sam3_video_predictor.py").read_text(encoding="utf-8")
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Sam3VideoPredictor"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "handle_request"
    ]

    class Base:
        def handle_request(self, request):
            return ("original", request)

        def add_prompt(self, **kwargs):
            return kwargs

    namespace = {"torch": torch, "Sam3BasePredictor": Base}
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "video_request", "exec"),
        namespace,
    )
    predictor = namespace["Sam3VideoPredictor"]()
    backbone = SimpleNamespace()
    predictor.model = SimpleNamespace(detector=SimpleNamespace(backbone=backbone))
    ordinary = {"type": "add_prompt", "text": "left hand"}
    assert predictor.handle_request(ordinary) == ("original", ordinary)
    backbone.learned_prompt = make_prompt()
    request = {
        "type": "add_learned_prompt",
        "session_id": "session",
        "frame_index": 0,
        "target_id": "left",
    }
    assert predictor.handle_request(request) == {
        "session_id": "session",
        "frame_idx": 0,
        "text": "learned:left",
    }
    with pytest.raises(ValueError, match="Loaded target"):
        predictor.handle_request(dict(request, target_id="right"))
    with pytest.raises(ValueError, match="add_learned_prompt"):
        predictor.handle_request(ordinary)
    points = {"type": "add_prompt", "points": [[0.5, 0.5]]}
    assert predictor.handle_request(points) == ("original", points)


def test_prepare_random_without_text_encoder(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_prompt", ROOT / "scripts/prepare_learned_prompt.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    monkeypatch.setitem(sys.modules, "sam3.model.learned_prompt", learned)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", None)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"hash only; no model loading")
    args = ["--checkpoint", str(checkpoint), "--target-id", "right_hand"]
    for count in (1, 4, 32):
        output = tmp_path / f"random_{count}.pt"
        options = args + ["--output", str(output), "--init", "random"]
        for invalid in ([], ["--num-tokens", "0"], ["--num-tokens", "33"]):
            with pytest.raises(SystemExit):
                prepare.parse_args(options + invalid)
        prepare.main(options + ["--num-tokens", str(count)])
        prompt = learned.LearnedPrompt.load(output, checkpoint)
        assert prompt.features.shape == (count, 256)
        assert torch.equal(prompt.padding_mask, torch.arange(32) >= count)
        expected = torch.randn((32, 256), generator=torch.Generator().manual_seed(0))
        assert torch.equal(prompt.full_features(), expected * 0.02)
        assert prompt.metadata["reference_text"] is None
        assert prompt.metadata["num_tokens"] == count
        with pytest.raises(SystemExit):
            prepare.main(options + ["--num-tokens", str(count)])
    output_args = args + ["--output", str(tmp_path / "unused.pt")]
    for invalid in (
        [],
        ["--reference-text", " "],
        ["--reference-text", "hand", "--num-tokens", "4"],
        ["--init", "random", "--num-tokens", "4", "--reference-text", "hand"],
    ):
        with pytest.raises(SystemExit):
            prepare.parse_args(output_args + invalid)
