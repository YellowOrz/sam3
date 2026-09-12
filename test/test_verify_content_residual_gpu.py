"""CPU contract tests only: these do not claim full SAM3 GPU equivalence."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from scripts import verify_content_residual_gpu as probe
from scripts.cached_ve_text_features import CachedVETextEncoder, CLASS_NAMES


def features():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[0, [0, 1, 3, 5]] = False
    padding[1, [0, 2, 4, 6]] = False
    resized = torch.ones(32, 2, 256, dtype=torch.bfloat16)
    raw = torch.ones(32, 2, 1024, dtype=torch.float32)
    encoder = CachedVETextEncoder(padding, resized, raw, mode="content_delta",
        metadata={"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64})
    return encoder, encoder(list(CLASS_NAMES))


class ContentEngineeringContractTest(unittest.TestCase):
    def test_bounded_one_process_probe_cannot_become_resume_or_large_training(self):
        valid = SimpleNamespace(residual_positions="content", resume=None, preflight_only=False,
            stop_after_step=2, batch_size_per_rank=3, learning_rate=1e-4, anchor_weight=0.)
        probe.validate_probe_args(valid, ranks=(0, 0, 1))
        for ranks in ((0, 0, 2), (1, 1, 2), (0, 1, 1)):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                probe.validate_probe_args(valid, ranks=ranks)
        for field, value in (("residual_positions", "all"), ("resume", "checkpoint.pt"),
                ("preflight_only", True), ("stop_after_step", None), ("stop_after_step", 5),
                ("stop_after_step", 0), ("stop_after_step", True), ("stop_after_step", 2.),
                ("batch_size_per_rank", 2), ("learning_rate", .001), ("anchor_weight", .01)):
            changed = deepcopy(valid)
            setattr(changed, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.validate_probe_args(changed, ranks=(0, 0, 1))

    def test_full_prediction_hash_equality_includes_values_shape_and_dtype(self):
        prediction = {name: torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
                      for name in probe.PREDICTION_KEYS}
        reference = probe.prediction_signatures(prediction)
        probe.assert_same_predictions(reference, probe.prediction_signatures(deepcopy(prediction)))
        for variation in ("value", "dtype", "shape", "missing", "nan"):
            changed = deepcopy(prediction)
            if variation == "value":
                changed["pred_masks"][0, 0, 0] = .01
            elif variation == "dtype":
                changed["pred_logits"] = changed["pred_logits"].bfloat16()
            elif variation == "shape":
                changed["pred_boxes"] = changed["pred_boxes"].reshape(2, 6)
            elif variation == "missing":
                del changed["presence_logit_dec"]
            else:
                changed["pred_masks"][0, 0, 0] = float("nan")
            with self.subTest(variation=variation), self.assertRaises((ValueError, RuntimeError)):
                probe.assert_same_predictions(reference, probe.prediction_signatures(changed))
        self.assertEqual(probe.tensor_signature(torch.tensor(1.))["shape"], [])

    def test_only_body_changes_are_permitted_while_all_four_valid_features_remain(self):
        encoder, reference = features()
        check = probe.verify_content_features(reference, encoder(list(CLASS_NAMES)),
            encoder.valid_positions, require_change=False)
        self.assertEqual(check["context_length"], 32)
        self.assertEqual(check["body_changed_elements_per_hand_position"], [[0, 0], [0, 0]])
        with torch.no_grad():
            encoder.delta.fill_(.125)
        after = encoder(list(CLASS_NAMES))
        check = probe.verify_content_features(reference, after, encoder.valid_positions, require_change=True)
        self.assertEqual(check["body_changed_elements_per_hand_position"], [[256, 256], [256, 256]])
        with self.assertRaises(RuntimeError):
            probe.verify_content_features(reference, after, encoder.valid_positions, require_change=False)
        with self.assertRaises(RuntimeError):
            probe.verify_content_features(reference, reference, encoder.valid_positions, require_change=True)

    def test_feature_probe_rejects_special_padding_raw_shape_and_dtype_changes(self):
        encoder, reference = features()
        reference = tuple(value.detach().clone() for value in reference)
        for variant in ("bos", "eos", "pad_feature", "mask", "raw", "length", "dtype", "positions"):
            after = [value.clone() for value in reference]
            positions = encoder.valid_positions.clone()
            if variant == "bos":
                after[1][positions[0, 0], 0, 0] += 1
            elif variant == "eos":
                after[1][positions[1, 3], 1, 0] += 1
            elif variant == "pad_feature":
                after[1][31, 0, 0] += 1
            elif variant == "mask":
                after[0][0, 0] = True
            elif variant == "raw":
                after[2][1, 0, 0] += 1
            elif variant == "length":
                after[1] = after[1][:4]
            elif variant == "dtype":
                after[1] = after[1].float()
            else:
                positions[0, 1] = 2
            with self.subTest(variant=variant), self.assertRaises((ValueError, RuntimeError)):
                probe.verify_content_features(reference, after, positions, require_change=False)


if __name__ == "__main__":
    unittest.main()
