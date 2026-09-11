"""CPU tests; synthetic documents only, not Windows runtime validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.package_review_docs import package
from scripts.publish_review_docs import publish


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        (self.repo / 'docs').mkdir(parents=True)
        (self.repo / 'docs/intro.md').write_text('# Synthetic fixture')
        self.results = self.root / 'results'
        self.results.mkdir()
        self.bundle = self.root / 'delivery' / 'v1'
        package(repo_root=self.repo, output_dir=self.bundle, reports=[], results_roots=[self.results])
        self.feed = self.bundle.parent / 'latest.json'

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_and_idempotent(self):
        first = publish(self.bundle, self.feed)
        self.assertEqual(first, publish(self.bundle, self.feed))
        self.assertEqual(first['format'], 'sam3-review-feed-v1')
        self.assertEqual(first['version'], 'v1')
        self.assertEqual(json.loads(self.feed.read_text()), first)

    def test_corrupt_archive_never_published(self):
        with (self.bundle / 'review-docs.zip').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            publish(self.bundle, self.feed)
        self.assertFalse(self.feed.exists())

    def test_reject_unrelated_destination(self):
        self.feed.write_text('{"format": "unrelated"}')
        with self.assertRaises(ValueError):
            publish(self.bundle, self.feed)
        self.assertEqual(json.loads(self.feed.read_text())['format'], 'unrelated')

    def test_reject_symlink(self):
        alternate = self.root / 'other.json'
        alternate.write_text('unchanged')
        self.feed.symlink_to(alternate)
        with self.assertRaises(ValueError):
            publish(self.bundle, self.feed)
        self.assertEqual(alternate.read_text(), 'unchanged')

    def test_reject_destination_outside_release_parent(self):
        with self.assertRaises(ValueError):
            publish(self.bundle, self.root / 'latest.json')

    def test_version_identity_immutable(self):
        feed = publish(self.bundle, self.feed)
        feed['sha256'] = '0' * 64
        self.feed.write_text(json.dumps(feed))
        with self.assertRaisesRegex(ValueError, 'identity'):
            publish(self.bundle, self.feed)

    def test_concurrent_archive_change_keeps_previous_feed(self):
        previous = publish(self.bundle, self.feed)
        real_zipfile = zipfile.ZipFile
        def changing_archive(*args, **kwargs):
            opened = real_zipfile(*args, **kwargs)
            with (self.bundle / 'review-docs.zip').open('ab') as stream:
                stream.write(b'concurrent-change')
            return opened
        with patch('scripts.publish_review_docs.zipfile.ZipFile', side_effect=changing_archive):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                publish(self.bundle, self.feed)
        self.assertEqual(json.loads(self.feed.read_text()), previous)


if __name__ == '__main__':
    unittest.main()
