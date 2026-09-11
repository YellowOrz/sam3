import io
from pathlib import Path
import tempfile
import unittest
import stat
import struct
import zipfile
import hashlib

from scripts.audit_egohos_sample import (
    ArchiveRanges, DATASET_ID, LIMIT, PublicProbe, RESERVE, choose_complete_train, decode_member,
    public_log_url, safe_member, validate_range, validate_url,
)


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, *args):
        self.read_calls += 1
        return super().read(*args)


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)

    def open(self, req, timeout):
        return next(self.responses)


class PublicProbeTests(unittest.TestCase):
    def test_range_requires_real_exact_206(self):
        self.assertEqual(validate_range("bytes=0-0", 206, {"content-range": "bytes 0-0/100"}), 100)
        for status, headers in [(200, {"content-range": "bytes 0-0/100"}),
                                (206, {"content-range": "bytes 1-1/100"}),
                                (206, {"content-range": "bytes 0-0/*"}),
                                (206, {"content-range": "bytes 0-0/100", "content-length": "2"}),
                                (206, {"content-range": "bytes 0-0/100", "content-encoding": "gzip"})]:
            with self.assertRaises(ValueError):
                validate_range("bytes=0-0", status, headers)

    def test_no_unknown_hosts_or_credentials(self):
        validate_url("https://drive.google.com/uc?id=abc")
        for url in ["http://drive.google.com/", "https://drive.google.com.evil.test/",
                    "https://user:secret@drive.google.com/", "https://127.0.0.1/",
                    "https://drive.google.com:444/"]:
            with self.assertRaises(ValueError):
                validate_url(url)

    def test_signed_redirect_values_not_logged(self):
        self.assertEqual(public_log_url("https://drive.google.com/uc?id=public&token=private&export=download"),
                         "https://drive.google.com/uc?id=public&export=download")

    def test_ignored_range_body_is_never_read(self):
        response = Response(b"entire big archive", headers={"Content-Length": "1000000000"})
        with tempfile.TemporaryDirectory() as directory:
            probe = PublicProbe(Path(directory), Opener([response]))
            try:
                with self.assertRaises(ValueError):
                    probe.fetch("https://drive.google.com/uc?id=x", method="GET", byte_range="bytes=0-0", max_bytes=1)
                self.assertEqual(response.read_calls, 0)
                self.assertEqual(probe.ledger["body_bytes_received"], 0)
                self.assertEqual(probe.ledger["budget_charged_bytes"], RESERVE)
            finally:
                probe.close()

    def test_exact_range_counts_and_saves_original(self):
        response = Response(b"P", 206, {"Content-Length": "1", "Content-Range": "bytes 0-0/100"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = PublicProbe(root, Opener([response]))
            try:
                result = probe.fetch("https://drive.google.com/uc?id=x", method="GET", byte_range="bytes=0-0", max_bytes=1, output="probe.bin")
                self.assertEqual((root / "probe.bin").read_bytes(), b"P")
                self.assertEqual(result["archive_total_bytes"], 100)
                self.assertEqual(probe.ledger["budget_charged_bytes"], RESERVE + 1)
            finally:
                probe.close()

    def test_exact_206_without_content_length(self):
        response = Response(b'P', 206, {'Content-Range': 'bytes 0-0/100'})
        with tempfile.TemporaryDirectory() as directory:
            probe = PublicProbe(Path(directory), Opener([response]))
            try:
                result = probe.fetch('https://drive.google.com/', method='GET', byte_range='bytes=0-0', max_bytes=1)
                self.assertEqual(result['received_bytes'], 1)
            finally:
                probe.close()

    def test_escape_and_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = PublicProbe(root, Opener([]))
            try:
                with self.assertRaises(ValueError):
                    probe.fetch("https://drive.google.com/", output="../outside")
                (root / "exists").touch()
                with self.assertRaises(FileExistsError):
                    probe.fetch("https://drive.google.com/", output="exists")
                self.assertEqual(len(probe.ledger["requests"]), 0)
            finally:
                probe.close()

    def test_redirect_to_unapproved_host_is_not_followed(self):
        response = Response(b"unused", 302, {"Location": "https://accounts.google.com/login"})
        with tempfile.TemporaryDirectory() as directory:
            probe = PublicProbe(Path(directory), Opener([response]))
            try:
                with self.assertRaises(ValueError):
                    probe.fetch("https://drive.google.com/")
                self.assertEqual(response.read_calls, 0)
                self.assertEqual(len(probe.ledger["requests"]), 1)
            finally:
                probe.close()

    def test_budget_refuses_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = PublicProbe(Path(directory), Opener([]))
            try:
                probe.ledger["reserved_transport_bytes"] = LIMIT - RESERVE
                with self.assertRaises(RuntimeError):
                    probe.fetch("https://drive.google.com/", method="GET", max_bytes=1)
                self.assertEqual(len(probe.ledger["requests"]), 0)
            finally:
                probe.close()

    def test_zip64_local_member_and_crc(self):
        raw = b'original payload for CRC verification'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            with archive.open('data/train/label/example.png', 'w', force_zip64=True) as handle:
                handle.write(raw)
        contents = stream.getvalue()
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            info = archive.infolist()[0]
        header = contents[info.header_offset:info.header_offset + 30]
        fields = struct.unpack('<4s5H3I2H', header)
        size = fields[-2] + fields[-1] + info.compress_size
        block = contents[info.header_offset + 30:info.header_offset + 30 + size]
        self.assertEqual(decode_member(info, header, block), raw)
        changed = bytearray(header)
        changed[14] ^= 1
        with self.assertRaises(ValueError):
            decode_member(info, changed, block)
        changed_block = bytearray(block)
        # force_zip64 local extra starts immediately after name; corrupt its first size.
        changed_block[fields[-2] + 4] ^= 1
        with self.assertRaises(ValueError):
            decode_member(info, header, changed_block)
        info.CRC ^= 1
        with self.assertRaises(ValueError):
            decode_member(info, header, block)

    def test_selection_is_content_independent_and_complete(self):
        entries = [zipfile.ZipInfo(f'data/train/{kind}/{index:02d}.png')
                   for index in range(30) for kind in ('image', 'label', 'contact')]
        entries.append(zipfile.ZipInfo('data/train/image/incomplete.jpg'))
        eligible, selected = choose_complete_train(entries, count=20, seed=20260911)
        self.assertEqual(len(eligible), 30)
        self.assertEqual(len(selected), 20)
        self.assertEqual([x[0] for x in selected], [x[0] for x in choose_complete_train(list(reversed(entries)), 20, 20260911)[1]])

    def test_missing_contact_kept_missing_not_negative(self):
        entries = [zipfile.ZipInfo(f'data/train/{kind}/{index:02d}.png')
                   for index in range(20) for kind in ('image', 'label')]
        eligible, selected = choose_complete_train(entries)
        self.assertEqual(len(eligible), 20)
        self.assertTrue(all(set(row) == {'image', 'label'} for _, row in selected))

    def test_same_stem_from_different_train_roots_cannot_pair(self):
        entries = [zipfile.ZipInfo('sourceA/train/image/same.jpg'),
                   zipfile.ZipInfo('sourceB/train/label/same.png')]
        with self.assertRaises(ValueError):
            choose_complete_train(entries, count=1)

    def test_png_directory_is_not_a_label(self):
        entries = [zipfile.ZipInfo('data/train/image/same.jpg'),
                   zipfile.ZipInfo('data/train/label/same.png/')]
        with self.assertRaises(ValueError):
            choose_complete_train(entries, count=1)

    def test_cache_is_bound_to_public_source_and_http_validator(self):
        url = f'https://drive.usercontent.google.com/download?id={DATASET_ID}&export=download'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'archive-ranges').mkdir()
            probe = PublicProbe(root, Opener([]))
            try:
                for position in (0, 1):
                    relative = f'archive-ranges/{position}-{position}.bin'
                    (root / relative).write_bytes(b'P')
                    probe.ledger['requests'].append({
                        'file': relative, 'outcome': 'complete', 'archive_total_bytes': 100,
                        'url': url, 'range': f'bytes={position}-{position}', 'received_bytes': 1,
                        'sha256': hashlib.sha256(b'P').hexdigest(),
                        'headers': {'etag': 'version1' if position == 0 else 'version2'},
                    })
                reader = ArchiveRanges(probe, url, 100)
                self.assertEqual(reader.block(0, 1), b'P')
                with self.assertRaises(ValueError):
                    reader.block(1, 1)
                reader = ArchiveRanges(probe, url.replace('usercontent.google.com/download', 'google.com/uc'), 100)
                with self.assertRaises(ValueError):
                    reader.block(0, 1)
            finally:
                probe.close()

    def test_member_paths_symlinks_and_encryption_rejected(self):
        for name in ('../escape.png', '/absolute.png', 'data\\escape.png'):
            with self.assertRaises(ValueError):
                safe_member(zipfile.ZipInfo(name))
        info = zipfile.ZipInfo('data/train/image/valid.png')
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            safe_member(info)
        info.external_attr = 0
        info.flag_bits = 1
        with self.assertRaises(ValueError):
            safe_member(info)


if __name__ == "__main__":
    unittest.main()
