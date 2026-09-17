"""Exact saved-prefix checks for storage-only video inference changes."""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SavedOutputEquivalence:
    def __init__(self, baseline, runinfo):
        self.root = Path(baseline)
        self.hashes = {'run.json': sha(self.root / 'run.json')}
        original = json.loads((self.root / 'run.json').read_text())
        if (original.get('tracker_policy', 'legacy') != runinfo.get('tracker_policy', 'legacy')
                or original.get('tracker_policy_sha256') != runinfo.get('tracker_policy_sha256')):
            raise ValueError('Algorithm policy differs; not a storage-only equivalence check')
        for key in ('contract', 'plan_sha256', 'base_sha256', 'model_source_sha256',
                    'mode', 'gpu', 'torch_version'):
            if original[key] != runinfo[key]:
                raise ValueError(f'Unpaired original output field: {key}')
        if original['engineering']:
            raise ValueError('Require original full-video output, not an engineering prefix')
        self.mode = original['mode']
        self.original_runner_sha256 = original['runner_sha256']
        self.rows, self.seen, self.expected = {}, set(), set()
        self.completed = []

    def begin(self, name, count):
        self.rows, self.seen, self.expected = {}, set(), set()
        path = self.root / f'{name}.jsonl'
        if not path.exists():
            self.active = dict(sequence=name, original_frames=0, executed_frames=count)
            return
        self.hashes[path.name] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        methods = (f'frame_{self.mode}', f'video_{self.mode}')
        if not rows or len(rows) % 2:
            raise ValueError('Original prefix has missing frame/method pairs')
        original_count = len(rows) // 2
        expected = {(i, method) for i in range(original_count) for method in methods}
        keys = [(r['frame_index'], r['method']) for r in rows]
        if (len(set(keys)) != len(keys) or set(keys) != expected
                or any(r['sequence'] != name for r in rows)):
            raise ValueError('Original prefix is not contiguous and paired from frame zero')
        marker = self.root / f'{name}-complete.json'
        if marker.exists():
            self.hashes[marker.name] = sha(marker)
            proof = json.loads(marker.read_text())
            if (proof['status'] != 'complete' or proof['frames'] != original_count
                    or proof['records_sha256'] != self.hashes[path.name]):
                raise ValueError('Original completed sequence changed')
        elif (self.root / 'failure.json').exists():
            self.hashes['failure.json'] = sha(self.root / 'failure.json')
        else:
            raise ValueError('Original prefix is neither completed nor from a stopped failed run')
        self.rows = {(r['frame_index'], r['method']): r for r in rows}
        self.expected = {(i, m) for i, m in expected if i < count}
        self.active = dict(sequence=name, original_frames=original_count, executed_frames=count)

    def check(self, row):
        if row['sequence'] != self.active['sequence']:
            raise ValueError('Comparison sequence differs')
        key = row['frame_index'], row['method']
        if key not in self.rows:
            return
        if key in self.seen:
            raise ValueError('Original-prefix output compared twice')
        if row != self.rows[key]:
            fields = sorted(k for k in set(row) | set(self.rows[key])
                            if row.get(k) != self.rows[key].get(k))
            raise ValueError(f'Storage-only output changed: {row["sequence"]} {key}: {fields}')
        self.seen.add(key)

    def finish_sequence(self):
        if self.seen != self.expected:
            raise ValueError('Did not compare every executed original-prefix output')
        self.completed.append(dict(self.active, identical_records=len(self.seen)))

    def summary(self):
        for name, digest in self.hashes.items():
            if sha(self.root / name) != digest:
                raise ValueError('Original outputs changed during paired comparison')
        return dict(status='all-available-original-prefixes-exact',
            original_runner_sha256=self.original_runner_sha256,
            source_sha256=self.hashes, sequences=self.completed,
            identical_records=sum(r['identical_records'] for r in self.completed))
