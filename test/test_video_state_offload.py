import inspect
import unittest

import torch

from scripts.eval.video_state_offload import (
    assert_tracker_state_cpu,
    force_forward_output_cache_cpu,
    force_tracker_state_cpu,
    tracker_state_storage_snapshot,
)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def init_state(self, width=640, cache=None, offload_state_to_cpu=False, *, flag=7):
        self.calls.append((width, cache, offload_state_to_cpu, flag))
        return {
            "storage_device": torch.device("cpu" if offload_state_to_cpu else "cuda"),
            "offload_state_to_cpu": offload_state_to_cpu,
        }


class FakeVideo:
    def _cache_frame_outputs(self, inference_state, frame_idx, obj_id_to_mask,
                             suppressed_obj_ids=None, removed_obj_ids=None,
                             unconfirmed_obj_ids=None):
        excluded = set(suppressed_obj_ids or ()) | set(removed_obj_ids or ())
        excluded.update(unconfirmed_obj_ids or ())
        inference_state["cached_frame_outputs"][frame_idx] = {
            key: value for key, value in obj_id_to_mask.items() if key not in excluded
        }
        return "sentinel"

    def _build_tracker_output(self, inference_state, frame_idx):
        return inference_state["cached_frame_outputs"][frame_idx].copy()


class VideoStateOffloadTests(unittest.TestCase):
    def test_opt_in_only_and_default_arguments_preserved(self):
        tracker, other = FakeTracker(), FakeTracker()
        original_class_method = FakeTracker.init_state
        original_signature = inspect.signature(tracker.init_state)
        self.assertEqual(str(tracker.init_state()["storage_device"]), "cuda")
        with force_tracker_state_cpu(tracker) as handle:
            self.assertEqual(inspect.signature(tracker.init_state), original_signature)
            state = tracker.init_state()
            self.assertEqual(str(state["storage_device"]), "cpu")
            self.assertEqual(tracker.calls[-1], (640, None, True, 7))
            self.assertEqual(handle.states_created, 1)
            self.assertEqual(str(other.init_state()["storage_device"]), "cuda")
            self.assertIs(FakeTracker.init_state, original_class_method)
        self.assertNotIn("init_state", vars(tracker))
        self.assertEqual(str(tracker.init_state()["storage_device"]), "cuda")
        handle.restore()  # Idempotent.

    def test_positional_and_keyword_values_preserved(self):
        tracker = FakeTracker()
        sentinel = object()
        with force_tracker_state_cpu(tracker):
            tracker.init_state(320, sentinel, True, flag=9)
            self.assertEqual(tracker.calls[-1], (320, sentinel, True, 9))
            tracker.init_state(width=800, cache=sentinel)
            self.assertEqual(tracker.calls[-1], (800, sentinel, True, 7))

    def test_explicit_false_conflict_does_not_call_original(self):
        tracker = FakeTracker()
        with force_tracker_state_cpu(tracker):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                tracker.init_state(offload_state_to_cpu=False)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                tracker.init_state(640, None, False)
            self.assertEqual(tracker.calls, [])

    def test_exception_exit_restores_original(self):
        tracker = FakeTracker()
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            with force_tracker_state_cpu(tracker):
                raise RuntimeError("inference failure")
        self.assertNotIn("init_state", vars(tracker))
        self.assertEqual(str(tracker.init_state()["storage_device"]), "cuda")

    def test_existing_instance_override_restored(self):
        tracker = FakeTracker()
        bound = tracker.init_state
        tracker.init_state = bound
        with force_tracker_state_cpu(tracker):
            tracker.init_state()
        self.assertIs(tracker.init_state, bound)

    def test_nested_install_rejected(self):
        tracker = FakeTracker()
        with force_tracker_state_cpu(tracker):
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                force_tracker_state_cpu(tracker)

    def test_storage_assertion_and_failed_call_restoration(self):
        class IgnoringTracker(FakeTracker):
            def init_state(self, offload_state_to_cpu=False):
                return {"storage_device": "cuda", "offload_state_to_cpu": True}

        tracker = IgnoringTracker()
        with self.assertRaisesRegex(RuntimeError, "did not enable"):
            with force_tracker_state_cpu(tracker):
                tracker.init_state()
        self.assertNotIn("init_state", vars(tracker))
        with self.assertRaises(RuntimeError):
            assert_tracker_state_cpu({"storage_device": "cpu"})

    def test_snapshot_deduplicates_views_and_ignores_nonoutput_caches(self):
        base = torch.zeros(4, 5)
        view = base[:1]
        mask = torch.zeros(8, 8, dtype=torch.bool)
        output = {"maskmem_features": base, "pred_masks": mask}
        state = {
            "obj_ids": [1],
            "storage_device": "cpu",
            "output_dict": {
                "cond_frame_outputs": {0: output},
                "non_cond_frame_outputs": {1: {"maskmem_features": view}},
            },
            "output_dict_per_obj": {0: {"cond_frame_outputs": {0: output}}},
            "temp_output_dict_per_obj": {},
            "cached_features": {0: torch.zeros(10000)},
        }
        result = tracker_state_storage_snapshot(
            {"tracker_inference_states": [state, state]}
        )
        self.assertEqual(result["tracker_state_count"], 1)
        self.assertEqual(result["frame_output_count"], 2)
        self.assertEqual(result["cpu"]["tensor_count"], 3)
        self.assertEqual(result["cpu"]["storage_count"], 2)
        self.assertEqual(result["cpu"]["storage_bytes"], 4 * 5 * 4 + 8 * 8)
        self.assertEqual(result["large_tracker_outputs"]["maskmem_features"]["cpu"]["storage_bytes"], 80)
        self.assertEqual(result["large_tracker_outputs"]["pred_masks"]["cpu"]["storage_bytes"], 64)
        self.assertEqual(result["cuda"]["storage_bytes"], 0)
        self.assertEqual(tracker_state_storage_snapshot(state), result)

    def test_snapshot_handles_no_trackers_and_rejects_unrecognized_state(self):
        result = tracker_state_storage_snapshot({"tracker_inference_states": []})
        self.assertEqual(result["tracker_state_count"], 0)
        self.assertEqual(result["cpu"]["storage_bytes"], 0)
        with self.assertRaises(ValueError):
            tracker_state_storage_snapshot({})

    def test_auxiliary_mask_scopes_deduplicate_combined_storage(self):
        mask = torch.ones(3, 5, dtype=torch.bool)
        state = {
            "output_dict": {"cond_frame_outputs": {0: {"pred_masks": mask}}},
            "mask_inputs_per_obj": {7: {0: mask[:1]}},
        }
        result = tracker_state_storage_snapshot({
            "tracker_inference_states": [state],
            "cached_frame_outputs": {0: {7: mask}},
        })
        self.assertEqual(result["cpu"]["storage_bytes"], 15)
        self.assertEqual(result["cached_frame_outputs"]["cpu"]["storage_bytes"], 15)
        self.assertEqual(result["mask_inputs_per_obj"]["cpu"]["storage_bytes"], 15)
        self.assertEqual(result["combined"]["cpu"]["storage_bytes"], 15)
        self.assertEqual(result["cached_frame_outputs"]["frame_count"], 1)

    def test_restore_wont_clobber_unrelated_patch(self):
        tracker = FakeTracker()
        handle = force_tracker_state_cpu(tracker)
        later_patch = lambda: None
        tracker.init_state = later_patch
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            handle.restore()
        self.assertIs(tracker.init_state, later_patch)

    def test_forward_cache_preserves_filtering_and_caller_mask_storage(self):
        model = FakeVideo()
        state = {"cached_frame_outputs": {}}
        raw = torch.tensor([[True, False], [False, True]])
        masks = {1: raw, 2: raw, 3: raw, 4: raw}
        class_method = FakeVideo._cache_frame_outputs
        with force_forward_output_cache_cpu(model) as handle:
            result = model._cache_frame_outputs(
                state, 9, masks, suppressed_obj_ids={2},
                removed_obj_ids={3}, unconfirmed_obj_ids={4},
            )
            self.assertEqual(result, "sentinel")
            self.assertEqual(handle.frames_cached, 1)
            self.assertEqual(set(state["cached_frame_outputs"][9]), {1})
            saved = state["cached_frame_outputs"][9][1]
            self.assertTrue(torch.equal(saved, raw))
            self.assertEqual(saved.dtype, raw.dtype)
            self.assertNotEqual(saved.untyped_storage().data_ptr(), raw.untyped_storage().data_ptr())
            self.assertEqual(set(masks), {1, 2, 3, 4})
            raw[0, 0] = False
            self.assertTrue(saved[0, 0].item())
            self.assertIs(FakeVideo._cache_frame_outputs, class_method)
            with self.assertRaisesRegex(RuntimeError, "noninteractive forward"):
                model._build_tracker_output(state, 9)
        self.assertNotIn("_cache_frame_outputs", vars(model))
        self.assertNotIn("_build_tracker_output", vars(model))
        self.assertEqual(set(model._build_tracker_output(state, 9)), {1})
        handle.restore()

    def test_cache_context_restores_on_error_and_preserves_float_values(self):
        model = FakeVideo()
        state = {"cached_frame_outputs": {}}
        mask = torch.tensor([[0.125, -3.75]], dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "failure"):
            with force_forward_output_cache_cpu(model):
                model._cache_frame_outputs(
                    inference_state=state, frame_idx=0, obj_id_to_mask={7: mask}
                )
                self.assertTrue(torch.equal(state["cached_frame_outputs"][0][7], mask))
                self.assertFalse(state["cached_frame_outputs"][0][7].requires_grad)
                raise ValueError("failure")
        self.assertNotIn("_cache_frame_outputs", vars(model))
        self.assertNotIn("_build_tracker_output", vars(model))

    def test_cache_rejects_nested_install_and_supports_empty_filtered_frame(self):
        model = FakeVideo()
        state = {"cached_frame_outputs": {}}
        with force_forward_output_cache_cpu(model):
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                force_forward_output_cache_cpu(model)
            model._cache_frame_outputs(state, 0, {})
            self.assertEqual(state["cached_frame_outputs"][0], {})


if __name__ == "__main__":
    unittest.main()
