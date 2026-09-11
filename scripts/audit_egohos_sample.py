"""Conservative public EgoHOS metadata/range probe; never fetch a full archive.

The probe uses no authentication, cookies, automatic retries or model inference.
It charges all read bodies plus one MiB per HTTP request against a 250 MiB cap.
Range failure closes the response without reading an archive body. The official
download scripts are evidence only; this module does not execute them.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import stat
import struct
import time
from urllib import error, parse, request
import zipfile
import zlib


LIMIT = 250 * 1024 * 1024
RESERVE = 1024 * 1024
DATASET_ID = "1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK"
DATASET_URL = f"https://drive.google.com/uc?export=download&id={DATASET_ID}"


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(url: str) -> None:
    parsed = parse.urlsplit(url)
    hosts = {
        "drive.google.com", "drive.usercontent.google.com",
        "raw.githubusercontent.com", "api.github.com",
        "www.seas.upenn.edu", "www.engineering.upenn.edu",
    }
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443)
            or not (host in hosts or host.endswith(".googleusercontent.com"))):
        raise ValueError("Only approved public HTTPS source hosts are permitted")


def public_log_url(url: str) -> str:
    """Keep public identity fields without retaining signed redirect credentials."""
    parsed = parse.urlsplit(url)
    query = [(key, value) for key, value in parse.parse_qsl(parsed.query)
             if key in {"id", "export", "recursive", "page", "per_page"}]
    return parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                            parse.urlencode(query), ""))


def validate_range(spec: str, status: int, headers: dict[str, str]) -> int:
    wanted = re.fullmatch(r"bytes=(\d+)-(\d+)", spec)
    returned = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("content-range", ""))
    if status != 206 or not wanted or not returned:
        raise ValueError("Real 206 with Content-Range required; full responses refused")
    start, end = map(int, wanted.groups())
    actual_start, actual_end, total = map(int, returned.groups())
    if start < 0 or end < start or (start, end) != (actual_start, actual_end) or end >= total:
        raise ValueError("Content-Range does not match requested offsets/total")
    if headers.get("content-encoding", "identity").lower() not in ("identity", ""):
        raise ValueError("Encoded range bodies are not accepted")
    length = headers.get("content-length")
    if length is not None and int(length) != end - start + 1:
        raise ValueError("Range Content-Length mismatch")
    return total


class PublicProbe:
    def __init__(self, root: Path, opener=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "network-ledger.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.root / "network-ledger.json"
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {
            "limit_bytes": LIMIT, "body_bytes_received": 0,
            "reserved_transport_bytes": 0, "requests": [],
        }
        if self.ledger["limit_bytes"] != LIMIT:
            raise ValueError("Unexpected prior budget")
        self.opener = opener or request.build_opener(NoRedirect())

    def close(self):
        self.lock.close()

    def save(self):
        self.ledger["budget_charged_bytes"] = (
            self.ledger["body_bytes_received"] + self.ledger["reserved_transport_bytes"]
        )
        self.path.write_text(json.dumps(self.ledger, ensure_ascii=False, indent=2) + "\n")

    def fetch(self, url, *, method="HEAD", byte_range=None, max_bytes=512 * 1024,
              output=None):
        if method not in ("HEAD", "GET") or max_bytes <= 0:
            raise ValueError("Invalid request")
        if output is not None:
            target = (self.root / output).resolve()
            target.relative_to(self.root)
            if target.exists():
                raise FileExistsError(target)
        else:
            target = None
        for _ in range(5):
            validate_url(url)
            if (self.ledger["body_bytes_received"] + self.ledger["reserved_transport_bytes"]
                    + RESERVE + (0 if method == "HEAD" else max_bytes) > LIMIT):
                raise RuntimeError("Conservative request allowance would exceed budget")
            item = {"url": public_log_url(url), "method": method, "range": byte_range,
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "received_bytes": 0}
            self.ledger["requests"].append(item)
            self.ledger["reserved_transport_bytes"] += RESERVE
            self.save()
            response = None
            try:
                headers = {"User-Agent": "EgoHOS-bounded-research-review/1.0",
                           "Accept-Encoding": "identity"}
                if byte_range:
                    headers["Range"] = byte_range
                req = request.Request(url, headers=headers, method=method)
                try:
                    response = self.opener.open(req, timeout=25)
                except error.HTTPError as exc:
                    response = exc
                item["status"] = response.status
                normalized = {k.lower(): v for k, v in response.headers.items()}
                # Do not store Set-Cookie values or signed redirect query strings.
                item["headers"] = {k: v for k, v in normalized.items() if k in {
                    "content-type", "content-length", "content-range", "accept-ranges",
                    "content-encoding", "etag", "last-modified", "content-disposition",
                }}
                if response.status in (301, 302, 303, 307, 308):
                    next_url = parse.urljoin(url, normalized["location"])
                    validate_url(next_url)
                    item["redirect_host"] = parse.urlsplit(next_url).hostname
                    item["outcome"] = "redirect_headers_only"
                    url = next_url
                    continue
                if method == "HEAD":
                    item["outcome"] = "metadata_only"
                    return item
                if byte_range:
                    item["archive_total_bytes"] = validate_range(byte_range, response.status, normalized)
                length = normalized.get("content-length")
                expected = int(length) if length is not None else None
                if expected is None and byte_range:
                    first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range).groups())
                    expected = last - first + 1
                if expected is not None and expected > max_bytes:
                    raise RuntimeError("Declared response exceeds per-request cap; body not read")
                body = bytearray()
                while len(body) < max_bytes:
                    chunk = response.read(min(65536, max_bytes - len(body)))
                    if not chunk:
                        break
                    body.extend(chunk)
                    item["received_bytes"] += len(chunk)
                    self.ledger["body_bytes_received"] += len(chunk)
                    self.save()
                if expected is None and len(body) == max_bytes:
                    raise RuntimeError("Unknown-length response reached cap; no additional read")
                if expected is not None and len(body) != expected:
                    raise RuntimeError("Incomplete response")
                if byte_range:
                    first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range).groups())
                    if len(body) != last - first + 1:
                        raise RuntimeError("Incomplete range")
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"HTTP {response.status}; bounded error body counted")
                item["sha256"] = hashlib.sha256(body).hexdigest()
                if target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        handle.write(body)
                    item["file"] = str(target.relative_to(self.root))
                item["outcome"] = "complete"
                return item
            except Exception as exc:
                item["outcome"] = "error"
                item["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if response is not None:
                    response.close()
                self.save()
        raise RuntimeError("Too many redirects")


class ArchiveRanges(io.RawIOBase):
    """Read-only seekable archive index using exact, persisted HTTP ranges."""
    def __init__(self, probe, url, total):
        self.probe, self.url, self.total = probe, url, total
        validate_url(url)
        self.public_url = public_log_url(url)
        query = parse.parse_qs(parse.urlsplit(url).query)
        if (query.get('id') != [DATASET_ID]
                or parse.urlsplit(url).hostname not in {'drive.google.com', 'drive.usercontent.google.com'}):
            raise ValueError('Archive URL is not the approved official dataset identity')
        self.validators = None
        self.position = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.total)
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def block(self, start, size):
        if start < 0 or size <= 0 or start + size > self.total or size > 24 * 1024 * 1024:
            raise ValueError("Archive range outside bounded limits")
        key = (start, size)
        if key not in self.cache:
            relative = f"archive-ranges/{start}-{start + size - 1}.bin"
            target = self.probe.root / relative
            if target.exists():
                prior = [row for row in self.probe.ledger['requests']
                         if row.get('file') == relative and row.get('outcome') == 'complete']
                if (not prior or prior[-1].get('archive_total_bytes') != self.total
                        or prior[-1].get('url') != self.public_url
                        or prior[-1].get('range') != f'bytes={start}-{start + size - 1}'
                        or prior[-1].get('received_bytes') != size
                        or prior[-1]['sha256'] != hashlib.sha256(target.read_bytes()).hexdigest()):
                    raise ValueError("Existing range is not bound to completed ledger")
                result = prior[-1]
            else:
                result = self.probe.fetch(self.url, method='GET',
                                          byte_range=f'bytes={start}-{start + size - 1}',
                                          max_bytes=size, output=relative)
                if result['archive_total_bytes'] != self.total:
                    raise ValueError("Archive total changed")
            validators = {name: result.get('headers', {}).get(name)
                          for name in ('etag', 'last-modified')}
            if self.validators is None:
                self.validators = validators
            elif validators != self.validators:
                raise ValueError('Archive HTTP validators changed across ranges')
            self.cache[key] = target.read_bytes()
        return self.cache[key]

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.total - self.position
        size = min(size, max(0, self.total - self.position))
        if size == 0:
            return b''
        data = self.block(self.position, size)
        self.position += len(data)
        return data


def safe_member(info):
    path = PurePosixPath(info.filename)
    if (info.is_dir() or path.is_absolute() or '..' in path.parts or '\\' in info.filename
            or stat.S_ISLNK(info.external_attr >> 16)
            or info.flag_bits & 1 or info.compress_type not in (0, 8)
            or not 0 <= info.file_size <= 20 * 1024 * 1024
            or not 0 <= info.compress_size <= 20 * 1024 * 1024):
        raise ValueError("Unsupported, unsafe, encrypted or oversized ZIP member")


def decode_member(info, header, block):
    safe_member(info)
    if len(header) != 30:
        raise ValueError('Incomplete fixed local ZIP header')
    fields = struct.unpack('<4s5H3I2H', header)
    if fields[0] != b'PK\x03\x04' or fields[2] != info.flag_bits or fields[3] != info.compress_type:
        raise ValueError("Local ZIP header differs from central directory")
    name_size, extra_size = fields[-2:]
    if len(block) != name_size + extra_size + info.compress_size:
        raise ValueError("Local ZIP member extent differs")
    name = block[:name_size].decode('utf-8' if info.flag_bits & 0x800 else 'cp437')
    if name != info.filename:
        raise ValueError("Local ZIP member name differs")
    extra = block[name_size:name_size + extra_size]
    offset, zip64 = 0, None
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise ValueError('Malformed local ZIP extra header')
        tag, length = struct.unpack('<HH', extra[offset:offset + 4])
        offset += 4
        if offset + length > len(extra):
            raise ValueError('Truncated local ZIP extra data')
        if tag == 1:
            if zip64 is not None:
                raise ValueError('Duplicate local ZIP64 extra field')
            zip64 = extra[offset:offset + length]
        offset += length
    if not info.flag_bits & 8:
        local_crc, local_compressed, local_uncompressed = fields[6:9]
        zip64_offset = 0
        resolved = []
        for value in (local_uncompressed, local_compressed):
            if value == 0xffffffff:
                if zip64 is None or zip64_offset + 8 > len(zip64):
                    raise ValueError('Missing local ZIP64 size value')
                value = struct.unpack('<Q', zip64[zip64_offset:zip64_offset + 8])[0]
                zip64_offset += 8
            resolved.append(value)
        if local_crc != info.CRC or resolved != [info.file_size, info.compress_size]:
            raise ValueError('Local ZIP CRC/sizes differ from central directory')
    payload = block[name_size + extra_size:]
    if info.compress_type == 8:
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(payload, info.file_size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Invalid or oversized deflated member")
    else:
        raw = payload
    if len(raw) != info.file_size or (zlib.crc32(raw) & 0xffffffff) != info.CRC:
        raise ValueError("ZIP member size/CRC32 mismatch")
    return raw


def choose_complete_train(entries, count=20, seed=20260911):
    groups = {}
    roots = set()
    for info in entries:
        path = PurePosixPath(info.filename)
        if len(path.parts) < 3 or path.parts[-3] != 'train':
            continue
        kind = path.parts[-2]
        if kind not in {'image', 'label', 'contact'} or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        safe_member(info)
        parent = str(path.parent.parent)
        roots.add(parent)
        row = groups.setdefault((parent, path.stem), {})
        if kind in row:
            raise ValueError("Ambiguous duplicate training sample identity")
        row[kind] = info
    if len(roots) != 1:
        raise ValueError('Ambiguous multiple train roots; do not cross-pair stems')
    groups = {stem: row for (_, stem), row in groups.items()}
    eligible = sorted(key for key, row in groups.items() if {'image', 'label'} <= set(row))
    if len(eligible) < count:
        raise ValueError("Insufficient complete official train image/label pairs")
    selected = sorted(random.Random(seed).sample(eligible, count))
    return eligible, [(key, groups[key]) for key in selected]


def review_archive(probe, url, total, count=20, seed=20260911):
    """Persist selection before fetching images; no inference or relabeling."""
    ranges = ArchiveRanges(probe, url, total)
    with zipfile.ZipFile(ranges, 'r', allowZip64=True) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate full ZIP member names")
        eligible, selected = choose_complete_train(entries, count, seed)
    plan = {
        'format': 'egohos-original-small-review-selection-v1',
        'official_dataset_id': DATASET_ID, 'archive_total_bytes_not_fetched': total,
        'seed': seed, 'sampling': 'Uniform without replacement from sorted complete official train image/label stems; contact optional and missing is not negative; no mask-content or prediction selection',
        'eligible_train_count': len(eligible),
        'eligible_stems_sha256': hashlib.sha256(('\n'.join(eligible) + '\n').encode()).hexdigest(),
        'archive_members': len(entries), 'selected': [],
    }
    for key, row in selected:
        plan['selected'].append({'sample_id': key, 'members': {
            kind: {'name': info.filename, 'crc32': info.CRC, 'compressed_bytes': info.compress_size,
                   'uncompressed_bytes': info.file_size, 'local_offset': info.header_offset}
            for kind, info in row.items()}})
    plan_path = probe.root / 'selection-plan.json'
    encoded = json.dumps(plan, indent=2) + '\n'
    if plan_path.exists() and plan_path.read_text() != encoded:
        raise ValueError("Existing fixed selection differs; no replacement sampling")
    if not plan_path.exists():
        plan_path.write_text(encoded)
    manifest = {'format': 'egohos-original-small-review-manifest-v1',
                'status': 'collecting', 'selected_count': count, 'samples': [],
                'selection_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
                'no_model_predictions': True, 'no_training': True}
    manifest_path = probe.root / 'manifest.json'
    for key, row in selected:
        sample = {'sample_id': key, 'official_split': 'train', 'files': {},
                  'contact_provided': 'contact' in row}
        for kind, info in row.items():
            header = ranges.block(info.header_offset, 30)
            fields = struct.unpack('<4s5H3I2H', header)
            size = fields[-2] + fields[-1] + info.compress_size
            block = ranges.block(info.header_offset + 30, size)
            raw = decode_member(info, header, block)
            destination = probe.root / 'originals' / kind / PurePosixPath(info.filename).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError("Existing original differs")
            if not destination.exists():
                destination.write_bytes(raw)
            sample['files'][kind] = {'path': str(destination.relative_to(probe.root)),
                                     'archive_member': info.filename,
                                     'sha256': hashlib.sha256(raw).hexdigest(),
                                     'bytes': len(raw), 'crc32_verified': True}
        manifest['samples'].append(sample)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('ORIGINAL_PAIR', len(manifest['samples']), key, flush=True)
    manifest['status'] = 'downloaded_pending_mask_review'
    manifest['network_body_bytes_received'] = probe.ledger['body_bytes_received']
    manifest['network_budget_charged_bytes'] = probe.ledger['budget_charged_bytes']
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def render_reviews(root):
    """Pillow previews derived from untouched integer GT labels, not inference."""
    import numpy as np
    from PIL import Image, ImageDraw
    root = Path(root)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    if len(manifest['samples']) != manifest['selected_count']:
        raise ValueError("Download incomplete; do not claim complete review")
    rows = []
    headings = ['Original RGB', 'Left reference (1)', 'Right reference (2)',
                'Object 1st (3-5)', 'Object 2nd (6-8)']
    for index, sample in enumerate(manifest['samples'], 1):
        inputs = {}
        for kind, record in sample['files'].items():
            raw = (root / record['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('Original changed before preview')
            with Image.open(io.BytesIO(raw)) as picture:
                picture.load()
                inputs[kind] = picture.copy()
        rgb = inputs['image'].convert('RGB')
        label = np.asarray(inputs['label'])
        contact = np.asarray(inputs['contact']) if 'contact' in inputs else None
        if label.ndim != 2:
            raise ValueError('Expected original 2D categorical GT raster')
        if rgb.size != inputs['label'].size:
            raise ValueError('Original RGB/GT dimensions disagree')
        values, counts = np.unique(label, return_counts=True)
        if not set(values.tolist()) <= set(range(9)):
            raise ValueError('Unexpected official class IDs; do not silently remap')
        contact_values = np.unique(contact).tolist() if contact is not None else None
        if contact is not None and (contact.shape != label.shape or not set(contact_values) <= {0, 1}):
            raise ValueError('Unexpected contact schema; preserve and inspect, do not guess')
        masks = {'left_gt': label == 1, 'right_gt': label == 2,
                 'object_first_order_union_gt': np.isin(label, [3, 4, 5]),
                 'object_second_order_union_gt': np.isin(label, [6, 7, 8])}
        directory = root / 'previews' / f'{index:02d}'
        directory.mkdir(parents=True, exist_ok=False)
        rgb.save(directory / 'rgb.png')
        panels = [rgb]
        for name, mask in masks.items():
            panel = Image.fromarray(mask.astype('uint8') * 255)
            panel.save(directory / f'{name}.png')
            panels.append(panel.convert('RGB'))
        if contact is not None:
            Image.fromarray((contact == 1).astype('uint8') * 255).save(directory / 'contact_region_reference.png')
        sample['raster_validation'] = {
            'width': rgb.width, 'height': rgb.height,
            'label_mode': inputs['label'].mode, 'label_dtype': str(label.dtype),
            'class_pixel_counts': {str(int(key)): int(value) for key, value in zip(values, counts)},
            'contact_original_values': contact_values,
            'contact_status': 'provided' if contact is not None else 'not provided; not a negative or an all-zero mask',
            'preview_directory': str(directory.relative_to(root)),
            'hand_forearm_glove_boundary': 'pending human review; not inferred from class IDs',
            'object_contact_interpretation': 'Objects are 1st/2nd-order semantic unions; contact is a separate region reference, not an object-instance mask',
        }
        row = Image.new('RGB', (5 * 384, 288 + 52), 'white')
        draw = ImageDraw.Draw(row)
        draw.text((8, 3), f'{index:02d}  {sample["sample_id"]}', fill='black')
        for col, panel in enumerate(panels):
            thumbnail = panel.copy()
            thumbnail.thumbnail((384, 288), Image.Resampling.NEAREST if col else Image.Resampling.LANCZOS)
            draw.text((col * 384 + 6, 23), headings[col], fill='black')
            row.paste(thumbnail, (col * 384 + (384 - thumbnail.width) // 2, 48))
        row.save(directory / 'comparison.png')
        rows.append(row)
    (root / 'contact-sheets').mkdir(exist_ok=False)
    for start in range(0, len(rows), 2):
        batch = rows[start:start + 2]
        sheet = Image.new('RGB', (1920, 340 * len(batch)), 'white')
        for index, row in enumerate(batch):
            sheet.paste(row, (0, index * 340))
        sheet.save(root / 'contact-sheets' / f'review-{start + 1:02d}-{start + len(batch):02d}.png')
    manifest['status'] = 'originals_verified_previews_ready_human_review_pending'
    manifest['mask_modifications'] = 'None; binary PNGs are derived display views only'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--get", action="store_true")
    parser.add_argument("--range", dest="byte_range")
    parser.add_argument("--max-bytes", type=int, default=512 * 1024)
    parser.add_argument("--output")
    args = parser.parse_args()
    probe = PublicProbe(args.output_root)
    try:
        result = probe.fetch(args.url, method="GET" if args.get else "HEAD",
                             byte_range=args.byte_range, max_bytes=args.max_bytes,
                             output=args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        probe.close()


if __name__ == "__main__":
    main()
