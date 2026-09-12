import copy
import io
import unittest

import torch
from torch import nn

from scripts.cached_ve_text_features import (
    CachedVETextEncoder, CLASS_NAMES, NATURAL_PROMPTS, capture_ve_text_cache,
    install_cached_ve_text_encoder, restore_original_ve_text_encoder,
    set_cached_ve_training_mode, expected_delta_shape, validate_delta_state,
)


def cache_inputs(dtype=torch.float32):
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[0, [0, 1, 3, 5]] = False
    padding[1, [0, 2, 4, 6]] = False
    generator = torch.Generator().manual_seed(73)
    resized = torch.randn(32, 2, 256, generator=generator).to(dtype)
    raw = torch.randn(32, 2, 1024, generator=generator, dtype=torch.float32)
    metadata = {"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64}
    return padding, resized, raw, metadata


def make_cache(mode="frozen", dtype=torch.float32):
    padding, resized, raw, metadata = cache_inputs(dtype)
    return CachedVETextEncoder(padding, resized, raw, metadata=metadata, mode=mode)


class FakeVE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(.8)
        padding, resized, raw, _ = cache_inputs(torch.bfloat16)
        self.register_buffer("padding", padding)
        self.register_buffer("resized", resized)
        self.register_buffer("raw", raw)
        self.calls = []

    def forward(self, texts, input_boxes=None, device=None):
        self.calls.append((tuple(texts), input_boxes, device))
        return self.padding, self.resized, self.raw


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.language_backbone = FakeVE()
        self.backbone.vision_backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.7))
        self.decoder = nn.Linear(4, 4)

    def forward(self, texts):
        return self.backbone.language_backbone(texts, device="cpu")


class CachedVEFeaturesTest(unittest.TestCase):
    def test_content_zero_delta_preserves_complete_triple_and_both_side_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                frozen = make_cache(dtype=dtype)
                all_positions = make_cache("zero_delta", dtype)
                content = make_cache("content_delta", dtype)
                names = ["right_hand", "left_hand", "right_hand"]
                actual = content(names)
                for reference in (frozen(names), all_positions(names)):
                    for value, expected in zip(actual, reference):
                        self.assertTrue(torch.equal(value, expected))
                        self.assertEqual(value.dtype, expected.dtype)
                self.assertEqual(tuple(content.delta.shape), (2, 2, 256))
                self.assertEqual(sum(value.numel() for value in content.parameters()), 1024)
                self.assertEqual(content.delta.dtype, torch.float32)
                actual[1].float().sum().backward()
                self.assertTrue(torch.equal(content.delta.grad[0], torch.ones(2, 256)))
                self.assertTrue(torch.equal(content.delta.grad[1], torch.full((2, 256), 2.)))
                self.assertTrue(all(not value.requires_grad and value.grad is None for value in content.buffers()))

    def test_content_delta_only_updates_middle_valid_positions_not_start_end_or_padding(self):
        content = make_cache("content_delta")
        before = content(list(CLASS_NAMES))
        buffers = {name: value.clone() for name, value in content.named_buffers()}
        with torch.no_grad():
            content.delta[0].fill_(.25)
            content.delta[1].fill_(.5)
        after = content(list(CLASS_NAMES))
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[2], after[2]))
        for side, delta in enumerate((.25, .5)):
            positions = content.valid_positions[side]
            selected = positions[1:3]
            untouched = torch.ones(32, dtype=torch.bool)
            untouched[selected] = False
            self.assertTrue(torch.equal(after[1][selected, side], before[1][selected, side] + delta))
            self.assertTrue(torch.equal(after[1][untouched, side], before[1][untouched, side]))
            self.assertTrue(torch.equal(after[1][positions[[0, 3]], side], before[1][positions[[0, 3]], side]))
            self.assertEqual(int((~after[0][side]).sum()), 4)
        for name, value in content.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]))

    def test_delta_shape_and_complete_state_validator_preserve_legacy_contract(self):
        for mode, shape in (("zero_delta", (2, 4, 256)), ("content_delta", (2, 2, 256))):
            with self.subTest(mode=mode):
                cache = make_cache(mode)
                state = cache.state_dict()
                self.assertEqual(expected_delta_shape(mode), shape)
                self.assertEqual(validate_delta_state(state), shape)
                self.assertEqual(set(state), {"delta", "padding_cache", "resized_cache", "raw_cache",
                                             "valid_positions", "_extra_state"})
                self.assertEqual(state["_extra_state"], {"mode": mode, "metadata": cache.cache_metadata})
        for invalid in ("frozen", "unknown", "", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_delta_shape(invalid)
        with self.assertRaises(ValueError):
            validate_delta_state(make_cache().state_dict())

    def test_content_state_validator_rejects_mode_shape_dtype_finiteness_and_position_corruption(self):
        source = make_cache("content_delta").state_dict()
        for variant in ("mode", "shape", "dtype", "nan", "positions", "position_dtype", "padding",
                        "metadata", "missing", "extra", "raw_nan"):
            state = copy.deepcopy(source)
            if variant == "mode":
                state["_extra_state"]["mode"] = "zero_delta"
            elif variant == "shape":
                state["delta"] = torch.zeros(2, 4, 256)
            elif variant == "dtype":
                state["delta"] = state["delta"].half()
            elif variant == "nan":
                state["delta"][0, 0, 0] = float("nan")
            elif variant == "positions":
                state["valid_positions"][0, 1] = 2
            elif variant == "position_dtype":
                state["valid_positions"] = state["valid_positions"].int()
            elif variant == "padding":
                state["padding_cache"][0, 0] = True
            elif variant == "metadata":
                state["_extra_state"]["metadata"]["prompt_texts"][0] = "left_hand"
            elif variant == "missing":
                del state["raw_cache"]
            elif variant == "extra":
                state["other"] = None
            else:
                state["raw_cache"][31, 1, 0] = float("nan")
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_delta_state(state)
        self.assertEqual(validate_delta_state(source), (2, 2, 256))

    def test_content_state_roundtrip_and_cross_mode_loading_is_rejected(self):
        content = make_cache("content_delta", torch.bfloat16)
        with torch.no_grad():
            content.delta.copy_(torch.linspace(-.25, .25, content.delta.numel()).reshape_as(content.delta))
        stream = io.BytesIO()
        torch.save(content.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        self.assertEqual(validate_delta_state(state), (2, 2, 256))
        restored = make_cache("content_delta", torch.bfloat16)
        restored.load_state_dict(state)
        self.assertEqual(restored.get_extra_state(), content.get_extra_state())
        for actual, expected in zip(restored(list(CLASS_NAMES)), content(list(CLASS_NAMES))):
            self.assertTrue(torch.equal(actual, expected))
        for source, target in (("content_delta", "zero_delta"), ("zero_delta", "content_delta"),
                               ("content_delta", "frozen"), ("frozen", "content_delta")):
            with self.subTest(source=source, target=target), self.assertRaises((ValueError, RuntimeError)):
                make_cache(target).load_state_dict(make_cache(source).state_dict())
        malformed = copy.deepcopy(state)
        malformed["delta"][0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            make_cache("content_delta", torch.bfloat16).load_state_dict(malformed)

    def test_content_capture_and_install_only_train_1024_parameters(self):
        original = FakeVE().eval()
        with torch.inference_mode():
            content = capture_ve_text_cache(original, base_checkpoint_sha256="a" * 64,
                tokenizer_sha256="b" * 64, mode="content_delta", device="cpu")
        self.assertFalse(torch.is_inference(content.delta))
        self.assertEqual(validate_delta_state(content.state_dict()), (2, 2, 256))
        model = FakeModel()
        base_state = {name: value.clone() for name, value in model.state_dict().items()
                      if not name.startswith("backbone.language_backbone.")}
        install_cached_ve_text_encoder(model, content)
        self.assertEqual(sum(value.numel() for value in model.parameters() if value.requires_grad), 1024)
        model(list(CLASS_NAMES))[1].float().sum().backward()
        self.assertTrue(torch.equal(content.delta.grad, torch.ones_like(content.delta)))
        for name, value in base_state.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value))

    def test_full_triple_order_repetition_and_mixed_dtypes_preserved(self):
        padding, resized, raw, metadata = cache_inputs(torch.bfloat16)
        cache = CachedVETextEncoder(padding, resized, raw, metadata=metadata)
        output = cache(["right_hand", "left_hand", "right_hand"], device="cpu")
        self.assertEqual(tuple(output[0].shape), (3, 32))
        self.assertEqual(tuple(output[1].shape), (32, 3, 256))
        self.assertEqual(tuple(output[2].shape), (32, 3, 1024))
        for actual, expected in zip(output, (padding[[1, 0, 1]], resized[:, [1, 0, 1]], raw[:, [1, 0, 1]])):
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(output[1].dtype, torch.bfloat16)
        self.assertEqual(output[2].dtype, torch.float32)
        self.assertEqual(sum(p.numel() for p in cache.parameters()), 0)

    def test_zero_delta_is_exactly_frozen_and_has_gradients_in_both_classes(self):
        frozen, learned = make_cache(dtype=torch.bfloat16), make_cache("zero_delta", torch.bfloat16)
        names = list(CLASS_NAMES)
        initial = learned(names)
        for actual, expected in zip(initial, frozen(names)):
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(sum(p.numel() for p in learned.parameters()), 2048)
        self.assertEqual(learned.delta.dtype, torch.float32)
        initial[1].float().sum().backward()
        self.assertTrue(torch.equal(learned.delta.grad, torch.ones_like(learned.delta)))
        self.assertTrue(all(not buffer.requires_grad and buffer.grad is None for buffer in learned.buffers()))

    def test_delta_changes_only_four_valid_resized_positions_never_raw_or_padding(self):
        cached = make_cache("zero_delta")
        before = cached(list(CLASS_NAMES))
        with torch.no_grad():
            cached.delta.fill_(.25)
        after = cached(list(CLASS_NAMES))
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[2], after[2]))
        for class_index in range(2):
            valid = ~cached.padding_cache[class_index]
            self.assertEqual(int(valid.sum()), 4)
            self.assertTrue(torch.equal(after[1][~valid, class_index], before[1][~valid, class_index]))
            self.assertTrue(torch.equal(after[1][valid, class_index], before[1][valid, class_index] + .25))

    def test_bad_shapes_nonfinite_padding_and_valid_count_rejected(self):
        for variant in ("short", "raw_dimension", "padding_dtype", "nan", "inf", "three_valid"):
            with self.subTest(variant=variant):
                padding, resized, raw, metadata = cache_inputs()
                if variant == "short":
                    resized = resized[:4]
                elif variant == "raw_dimension":
                    raw = raw[:, :, :256]
                elif variant == "padding_dtype":
                    padding = padding.float()
                elif variant == "nan":
                    resized[31, 0, 0] = float("nan")
                elif variant == "inf":
                    raw[31, 0, 0] = float("inf")
                else:
                    padding[0, 0] = True
                with self.assertRaises(ValueError):
                    CachedVETextEncoder(padding, resized, raw, metadata=metadata)

    def test_unknown_classes_natural_strings_and_placeholders_rejected(self):
        cache = make_cache()
        for captions in ([], "left_hand", ["left hand"], ["object"], ["visual"], ["<text_placeholder>"], [1]):
            with self.subTest(captions=captions), self.assertRaises(ValueError):
                cache(captions)
        with self.assertRaises(ValueError):
            cache(["left_hand"], input_boxes=[torch.zeros(4)])
        cache(["left_hand"], input_boxes=[])

    def test_metadata_required_immutable_and_matches_natural_prompts(self):
        padding, resized, raw, metadata = cache_inputs()
        for replacement in ({}, {**metadata, "tokenizer_sha256": "not-a-sha"},
                            {**metadata, "prompt_texts": ["left_hand", "right_hand"]},
                            {**metadata, "valid_positions": [[0, 1, 2, 3]] * 2}):
            with self.assertRaises(ValueError):
                CachedVETextEncoder(padding, resized, raw, metadata=replacement)
        cache = make_cache()
        copied = cache.cache_metadata
        copied["prompt_texts"][0] = "object"
        self.assertEqual(cache.cache_metadata["prompt_texts"], list(NATURAL_PROMPTS))

    def test_explicit_device_move_preserves_dtype_and_dtype_conversion_rejected(self):
        cache = make_cache("zero_delta", torch.bfloat16).to(device="cpu")
        self.assertEqual(cache(["left_hand"], device=torch.device("cpu"))[2].dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "Move CachedVETextEncoder explicitly"):
            cache(["left_hand"], device="meta")
        cache.to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "Cache dtype changed"):
            cache(["left_hand"])

    def test_capture_uses_eval_original_natural_prompts_and_preserves_all_outputs(self):
        original = FakeVE()
        args = {"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "device": "cpu"}
        with self.assertRaisesRegex(ValueError, "eval mode"):
            capture_ve_text_cache(original, **args)
        original.eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cached = capture_ve_text_cache(original, **args)
        self.assertEqual(original.calls, [(NATURAL_PROMPTS, None, "cpu")])
        for actual, expected in zip(cached(list(CLASS_NAMES)), (original.padding, original.resized, original.raw)):
            self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(cached.raw_cache.data_ptr(), original.raw.data_ptr())
        self.assertIn("full SAM3 GPU equivalence", cached.cache_metadata["equivalence_scope"])
        original.dropout.train()
        with self.assertRaises(ValueError):
            capture_ve_text_cache(original, **args)

    def test_install_keeps_vision_decoder_identity_and_only_delta_trainable(self):
        model = FakeModel()
        vision, decoder, original = model.backbone.vision_backbone, model.decoder, model.backbone.language_backbone
        old_weights = {name: value.clone() for name, value in model.state_dict().items()
                       if not name.startswith("backbone.language_backbone.")}
        cached = make_cache("zero_delta")
        self.assertIs(install_cached_ve_text_encoder(model, cached), original)
        self.assertIs(model.backbone.vision_backbone, vision)
        self.assertIs(model.decoder, decoder)
        self.assertEqual({name for name, value in model.named_parameters() if value.requires_grad},
                         {"backbone.language_backbone.delta"})
        self.assertTrue(all(not module.training for module in model.modules()))
        for name, value in old_weights.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value))
        model(list(CLASS_NAMES))[1].sum().backward()
        self.assertIsNotNone(cached.delta.grad)
        self.assertTrue(all(parameter.grad is None for parameter in vision.parameters()))
        model.train()
        with self.assertRaisesRegex(RuntimeError, "frozen-base path requires eval mode"):
            model(list(CLASS_NAMES))
        self.assertEqual(set_cached_ve_training_mode(model, train_delta=True), {"backbone.language_backbone.delta"})
        model(list(CLASS_NAMES))
        self.assertEqual(set_cached_ve_training_mode(model, train_delta=False), set())
        self.assertIs(restore_original_ve_text_encoder(model, original), cached)
        self.assertIs(model.backbone.language_backbone, original)
        self.assertFalse(hasattr(model, "_cached_ve_eval_guard_handle"))
        install_cached_ve_text_encoder(model, make_cache())
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_state_dict_round_trip_retains_metadata_buffers_and_delta(self):
        cache = make_cache("zero_delta", torch.bfloat16)
        with torch.no_grad():
            cache.delta.fill_(.125)
        stream = io.BytesIO()
        torch.save(cache.state_dict(), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        restored = make_cache("zero_delta", torch.bfloat16)
        restored.load_state_dict(saved)
        self.assertEqual(restored.cache_metadata, cache.cache_metadata)
        for actual, expected in zip(restored(list(CLASS_NAMES)), cache(list(CLASS_NAMES))):
            self.assertTrue(torch.equal(actual, expected))
        corrupt = copy.deepcopy(saved)
        corrupt["_extra_state"]["mode"] = "frozen"
        with self.assertRaises(ValueError):
            restored.load_state_dict(corrupt)

    def test_inference_mode_capture_still_allows_later_delta_gradients(self):
        original = FakeVE().eval()
        with torch.inference_mode():
            cache = capture_ve_text_cache(original, base_checkpoint_sha256="a" * 64,
                                         tokenizer_sha256="b" * 64, mode="zero_delta", device="cpu")
        self.assertFalse(torch.is_inference(cache.delta))
        self.assertTrue(all(not torch.is_inference(value) for value in cache.buffers()))
        cache(list(CLASS_NAMES))[1].float().sum().backward()
        self.assertTrue(torch.equal(cache.delta.grad, torch.ones_like(cache.delta)))


if __name__ == "__main__":
    unittest.main()
