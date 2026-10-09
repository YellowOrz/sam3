import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sam3.model import sam3_video_base
from sam3.model.sam3_video_base import (
    _associate_det_trk_compilable,
    LazyAssociateDetTrkResult,
    Sam3VideoBase,
)


@pytest.fixture
def multiplex_base():
    # Importing the GPU module queries CUDA hardware; execute its real methods on CPU.
    path = Path(sam3_video_base.__file__).with_name("sam3_multiplex_base.py")
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Sam3MultiplexBase"
    )
    names = {"_associate_det_trk", "_process_hotstart", "_process_hotstart_gpu"}
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = vars(sam3_video_base).copy()
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return SimpleNamespace(**{name: namespace[name] for name in names})


def associate(det, trk, use_iom=False, keep=None):
    return _associate_det_trk_compilable(
        det_masks=det,
        det_scores=torch.full((len(det),), 0.95),
        det_keep=torch.ones(len(det), dtype=torch.bool) if keep is None else keep,
        trk_masks=trk,
        new_det_thresh=0.7,
        iou_threshold_trk=0.5,
        iou_threshold=0.1,
        HIGH_CONF_THRESH=0.8,
        use_iom_recondition=use_iom,
        o2o_matching_masklets_enable=False,
        iom_thresh_recondition=0.8,
        iou_thresh_recondition=0.8,
    )


@pytest.mark.parametrize("use_iom", [False, True])
@pytest.mark.parametrize("det_count,trk_count", [(1, 2), (2, 1)])
def test_ambiguous_matches_reach_lifecycle_but_not_reconditioning(
    use_iom, det_count, trk_count
):
    result = LazyAssociateDetTrkResult(
        *associate(torch.ones(det_count, 4, 4), torch.ones(trk_count, 4, 4), use_iom)
    )
    assert not result.trk_is_unmatched.any()
    assert not result.is_new_det.any()
    assert result.im_mask.all()
    assert not result.det_is_high_iou.any()
    ids = np.arange(trk_count) + 10
    result._convert_to_numpy()
    realized = result._create_cpu_metadata(ids, torch.ones(det_count, 4, 4))
    for matched in realized.det_to_matched_trk_obj_ids.values():
        np.testing.assert_array_equal(matched, ids)
    assert realized.trk_id_to_max_iou_high_conf_det == {}


def test_unique_matches_still_recondition_and_dropped_detections_do_not_match():
    det = torch.full((2, 4, 4), -1.0)
    det[0, :2] = 1
    det[1, 2:] = 1
    trk = torch.cat((det, torch.zeros(1, 4, 4)))  # Padded tracker slot.
    result = LazyAssociateDetTrkResult(*associate(det, trk))
    assert result.im_mask.tolist() == [[True, False, False], [False, True, False]]
    assert result.det_is_high_iou.tolist() == [True, True]
    assert result.det_to_max_iou_trk_idx.tolist() == [0, 1]
    result = LazyAssociateDetTrkResult(
        *associate(det, trk, keep=torch.tensor([True, False]))
    )
    assert result.im_mask.tolist() == [[True, False, False], [False, False, False]]
    assert result.trk_is_unmatched.tolist() == [False, True, False]
    assert result.det_is_high_iou.tolist() == [True, False]


@pytest.mark.parametrize("multiplex", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_evidence_removes_newer_track_within_hotstart(
    multiplex, reverse, multiplex_base
):
    model = SimpleNamespace(
        assoc_iou_thresh=0.1,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.7,
        max_num_objects=4,
        use_iom_recondition=multiplex,
        o2o_matching_masklets_enable=False,
        iom_thresh_recondition=0.8,
        iou_thresh_recondition=0.8,
        hotstart_delay=15,
        hotstart_dup_thresh=2,
        hotstart_unmatch_thresh=8,
        init_trk_keep_alive=30,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        decrease_trk_keep_alive_for_empty_masklets=False,
        suppress_unmatched_only_within_hotstart=True,
    )
    det, trk = torch.ones(1, 4, 4), torch.ones(2, 4, 4)
    ids = np.array([10, 20])
    if multiplex:
        result = multiplex_base._associate_det_trk(
            model, det, torch.tensor([0.95]), torch.tensor([True]), trk, ids
        )
        assert result.im_mask.shape == (1, 2)  # Padded columns must not leak.
        assert result.im_mask.all() and not result.det_is_high_iou.any()
        result._convert_to_numpy()
        realized = result._create_cpu_metadata(ids, det)
        matches = realized.det_to_matched_trk_obj_ids
        assert realized.trk_id_to_max_iou_high_conf_det == {}
    else:
        _, _, matches, recondition, _ = Sam3VideoBase._associate_det_trk(
            model, det, np.array([0.95]), trk, ids
        )
        assert recondition == {}
    np.testing.assert_array_equal(matches[0], ids)
    metadata = {
        "obj_first_frame_idx": {10: 100 if reverse else 0, 20: 50},
        "unmatched_frame_inds": defaultdict(list),
        "trk_keep_alive": {10: 30, 20: 30},
        "overlap_pair_to_frame_inds": defaultdict(list),
        "removed_obj_ids": set(),
        "suppressed_obj_ids": defaultdict(set),
    }
    hotstart = (
        multiplex_base._process_hotstart
        if multiplex
        else Sam3VideoBase._process_hotstart
    )
    for index, frame in enumerate([49, 48] if reverse else [51, 52]):
        removed, metadata = hotstart(
            model,
            frame_idx=frame,
            num_frames=101,
            reverse=reverse,
            det_to_matched_trk_obj_ids=matches,
            new_det_obj_ids=np.array([], dtype=np.int64),
            empty_trk_obj_ids=np.array([], dtype=np.int64),
            unmatched_trk_obj_ids=np.array([], dtype=np.int64),
            rank0_metadata=metadata,
            tracker_metadata={},
        )
        assert removed == (set() if index == 0 else {20})
    assert metadata["overlap_pair_to_frame_inds"][(10, 20)] == (
        [49, 48] if reverse else [51, 52]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_multiplex_tensor_hotstart_counts_raw_duplicate_matches(
    reverse, multiplex_base, monkeypatch
):
    # Only host-to-device staging needs pinning; test the tensor bookkeeping on CPU.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    model = SimpleNamespace(
        hotstart_delay=15,
        hotstart_dup_thresh=2,
        hotstart_unmatch_thresh=8,
        min_trk_keep_alive=-1,
        max_trk_keep_alive=30,
        decrease_trk_keep_alive_for_empty_masklets=False,
        suppress_unmatched_only_within_hotstart=True,
    )
    result = LazyAssociateDetTrkResult(
        *associate(torch.ones(1, 4, 4), torch.ones(2, 4, 4))
    )
    metadata = {
        "N_obj": 2,
        "obj_first_frame": torch.tensor([100 if reverse else 0, 50]),
        "consecutive_unmatch_count": torch.zeros(2, dtype=torch.long),
        "trk_keep_alive": torch.full((2,), 30, dtype=torch.long),
        "removed_mask": torch.zeros(2, dtype=torch.bool),
        "overlap_pair_counts": torch.zeros(2, 2, dtype=torch.long),
        "last_occluded_tensor": torch.full((2,), -1, dtype=torch.long),
    }
    for index, frame in enumerate([49, 48] if reverse else [51, 52]):
        removed, suppressed, metadata = multiplex_base._process_hotstart_gpu(
            model, frame, reverse, result, {}, metadata
        )
        assert metadata["overlap_pair_counts"][0, 1].item() == index + 1
        assert removed.tolist() == [False, index == 1]
        assert not suppressed.any()
        assert not metadata["consecutive_unmatch_count"].any()


@pytest.mark.parametrize("use_iom", [False, True])
def test_compiled_association_retains_duplicate_evidence(use_iom):
    compiled = torch.compile(
        _associate_det_trk_compilable, backend="eager", fullgraph=True
    )
    result = LazyAssociateDetTrkResult(
        *compiled(
            torch.ones(1, 4, 4),
            torch.tensor([0.95]),
            torch.tensor([True]),
            torch.ones(2, 4, 4),
            0.7,
            0.5,
            0.1,
            0.8,
            use_iom,
            False,
            0.8,
            0.8,
        )
    )
    assert result.im_mask.tolist() == [[True, True]]
    assert not result.det_is_high_iou.any()
