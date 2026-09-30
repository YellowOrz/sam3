import torch

from sam3.model.sam3_video_base import clip_oversegmented_tracker_masks


def _masks(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_clip_replaces_tracker_that_contains_a_high_score_detection():
    det = _masks([[[5, 5, -5], [-5, -5, -5]]])
    trk = _masks([[[9, 9, 9], [9, 9, -5]]])
    scores = torch.tensor([0.9])

    clipped = clip_oversegmented_tracker_masks(det, scores, trk)

    assert torch.equal(clipped, det)


def test_clip_keeps_tracker_when_areas_already_agree():
    det = _masks([[[5, 5, -5], [-5, -5, -5]]])
    trk = _masks([[[4, 4, -4], [-4, -4, -4]]])
    original = trk.clone()

    clipped = clip_oversegmented_tracker_masks(det, torch.tensor([0.95]), trk)

    assert torch.equal(clipped, original)


def test_clip_skips_low_score_ambiguous_and_tiny_detections():
    hand = torch.zeros(6, 6)
    hand[:3, :3] = 5
    arm = hand.clone()
    arm[:, :] = 5
    speck = torch.zeros(6, 6)
    speck[0, 0] = 5
    outside = torch.zeros(6, 6)
    outside[5, 5] = 5
    outside[5, 4] = 5

    low_score = clip_oversegmented_tracker_masks(
        hand.unsqueeze(0), torch.tensor([0.4]), arm.unsqueeze(0).clone()
    )
    assert torch.equal(low_score, arm.unsqueeze(0))

    two = clip_oversegmented_tracker_masks(
        torch.stack([hand, hand]),
        torch.tensor([0.9, 0.95]),
        arm.unsqueeze(0).clone(),
    )
    assert torch.equal(two, arm.unsqueeze(0))

    tiny = clip_oversegmented_tracker_masks(
        speck.unsqueeze(0), torch.tensor([0.99]), arm.unsqueeze(0).clone()
    )
    assert torch.equal(tiny, arm.unsqueeze(0))

    uncovered = clip_oversegmented_tracker_masks(
        outside.unsqueeze(0), torch.tensor([0.99]), hand.unsqueeze(0).clone()
    )
    assert torch.equal(uncovered, hand.unsqueeze(0))


def test_clip_noop_on_empty_inputs():
    trk = torch.zeros(1, 2, 2)
    empty = torch.zeros(0, 2, 2)
    assert clip_oversegmented_tracker_masks(empty, torch.zeros(0), trk) is trk
    empty_trk = torch.zeros(0, 2, 2)
    assert clip_oversegmented_tracker_masks(trk, torch.ones(1), empty_trk).shape == (
        0,
        2,
        2,
    )
