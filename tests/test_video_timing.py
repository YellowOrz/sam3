"""CPU checks for optional stage timing; the callback never runs per frame."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
from sam3 import video_backend as backend, video_outputs


@pytest.mark.parametrize('frames,start', [(1, 0), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize('failure', [None, 'sam3', 'encoding'])
def test_phase_boundaries_include_normalization_and_always_close(tmp_path, monkeypatch, frames, start, failure):
    clock = [0.]
    monkeypatch.setattr(backend.time, 'monotonic', lambda: clock[0])
    events, requests = [], []
    class Predictor:
        def handle_request(self, body):
            kind = body['type']; requests.append(kind)
            clock[0] += 2
            return {'session_id': 'test'} if kind == 'start_session' else {'outputs': {}}
        def handle_stream_request(self, body):
            for index in range(frames):
                clock[0] += 3
                yield {'frame_index': index, 'outputs': {}}
    def normalize(_):
        clock[0] += 1
        if failure == 'sam3':
            raise RuntimeError('normalization failed')
        return {}, {}, {}
    def write(state, **kwargs):
        assert len(state.cache) == frames
        clock[0] += 11
        if failure == 'encoding':
            raise RuntimeError('encoding failed')
    monkeypatch.setattr(video_outputs, 'normalize_frame_outputs', normalize)
    monkeypatch.setattr(video_outputs, 'write_interactive_outputs', write)
    def call():
        backend._segment_clip(Predictor(), tmp_path/'rgb', video_outputs.VideoInfo(10, 8, frames, 30),
            tmp_path, tmp_path/'out', 'hand', 0, None, lambda *_: None, start_frame_index=start,
            phase=lambda *args: events.append(args))
    if failure:
        with pytest.raises(RuntimeError): call()
    else:
        call()
    assert requests[-1] == 'close_session'
    assert events[0][0] == 'sam3' and events[-1][0] == 'cleanup'
    assert events[-1][2] - events[-1][1] == 2
    assert events[0][3] is (failure != 'sam3')
    if failure != 'sam3':
        assert events[1][0] == 'encoding' and events[1][2] - events[1][1] == 11
        assert events[1][3] is (failure != 'encoding')
    assert sum(end - begin for _, begin, end, _ in events) == clock[0]


@pytest.mark.parametrize('chunk', [0, 1, 2, 3])
def test_batch_timing_counts_phases_not_frames_and_preserves_prompt_workload(tmp_path, monkeypatch, chunk):
    fake_torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(),
        cuda=SimpleNamespace(is_bf16_supported=lambda: False), bfloat16='bfloat16', float16='float16')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(video_outputs, 'probe_video', lambda _: video_outputs.VideoInfo(10, 8, 3, 30))
    monkeypatch.setattr(video_outputs, 'extract_frames', lambda *a: 3)
    monkeypatch.setattr(video_outputs, 'make_chunk_frame_dir', lambda *a: None)
    monkeypatch.setattr(video_outputs, 'merge_chunk_outputs', lambda *a, **k: None)
    monkeypatch.setattr(video_outputs, 'normalize_frame_outputs', lambda _: ({}, {}, {}))
    monkeypatch.setattr(video_outputs, 'write_interactive_outputs', lambda *a, **k: None)
    class Predictor:
        def handle_request(self, body):
            return {'session_id': 'test'} if body['type'] == 'start_session' else {'outputs': {}}
        def handle_stream_request(self, body):
            for index in range(body['max_frame_num_to_track'] + 1):
                yield {'frame_index': index, 'outputs': {}}
    events, prompts = [], []
    result = backend.segment_video(Predictor(), video=tmp_path/'rgb', prompts=['hand', 'cup'],
        output_dirs=[tmp_path/'hand', tmp_path/'cup'], chunk_frames=chunk,
        phase=lambda *a: events.append(a), prompt_done=lambda *a: prompts.append(a))
    blocks = 1 if not chunk else (3 + chunk - 1)//chunk
    assert result['frames'] == 3 and result['prompt_count'] == len(prompts) == 2
    assert sum(stage == 'sam3' for stage, *_ in events) == blocks * 2
    assert sum(stage == 'decoding' for stage, *_ in events) == 2
    assert all(complete for _, _, _, complete in events)
