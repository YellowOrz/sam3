import copy
import io
import json
import unittest
from collections import defaultdict

import numpy as np

from scripts.eval.trace_mano_video import LifecycleTrace, metadata_snapshot, mask_overlap_matrix, plain


class TraceTests(unittest.TestCase):
    def test_snapshot_does_not_create_missing_defaultdict_entry(self):
        metadata = dict(rank0_metadata=dict(suppressed_obj_ids=defaultdict(set),
            unmatched_frame_inds={3: [1, 2]}, trk_keep_alive={3: -1}),
            obj_id_to_tracker_score_frame_wise=defaultdict(dict))
        before = copy.deepcopy(metadata)
        result = metadata_snapshot(metadata, 9)
        self.assertEqual(metadata, before)
        self.assertEqual(result["rank0"]["unmatched_counts"], {"3": 2})
        metadata["rank0_metadata"]["trk_keep_alive"][3] = 5
        self.assertEqual(result["rank0"]["trk_keep_alive"], {"3": -1})

    def test_serialization_copies_numpy_and_sets(self):
        x = {3: np.array([4, 2]), "set": {5, 1}, "tuple": (np.int64(6),)}
        y = plain(x)
        x[3][0] = 99
        self.assertEqual(y, {"3": [4, 2], "set": [1, 5], "tuple": [6]})

    def test_overlap_known_values_and_empty_stacks(self):
        a = np.array([[[1, 1], [0, 0]]], bool)
        b = np.array([[[1, 0], [0, 0]], [[0, 0], [1, 1]]], bool)
        self.assertEqual(mask_overlap_matrix(a, b), [[.5, 0.]])
        self.assertEqual(mask_overlap_matrix(a, np.zeros((0, 2, 2), bool)), [[]])
        with self.assertRaises(ValueError):
            mask_overlap_matrix(a, np.zeros((1, 2, 3)))

    def test_tensor_metadata_is_copied_and_detached(self):
        import torch
        value = torch.tensor([.5, .75], requires_grad=True)
        result = plain({"scores": value})
        self.assertEqual(result, {"scores": [.5, .75]})
        self.assertTrue(value.requires_grad)
        self.assertIsNone(value.grad)

    def test_hotstart_wrapper_preserves_result_and_arguments(self):
        class Model:
            def _process_hotstart(self, frame_idx, rank0_metadata, new_det_obj_ids):
                rank0_metadata["trk_keep_alive"][3] -= 1
                return self.expected, rank0_metadata
        model = Model()
        model.expected = {7}
        stream = io.StringIO()
        trace = LifecycleTrace(model, stream)
        original = model._process_hotstart
        wrapped = trace._wrap("_process_hotstart", original)
        data = dict(trk_keep_alive={3: 1}, removed_obj_ids=set(), suppressed_obj_ids=defaultdict(set))
        out = wrapped(42, data, np.array([3]))
        self.assertIs(out[0], model.expected)
        self.assertIs(out[1], data)
        event = json.loads(stream.getvalue())
        self.assertEqual(event["keep_before"], {"3": 1})
        self.assertEqual(event["keep_after"], {"3": 0})
        self.assertEqual(dict(data["suppressed_obj_ids"]), {})


if __name__ == "__main__":
    unittest.main()
