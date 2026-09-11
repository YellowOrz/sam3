"""CPU fixtures for bounded, non-mutating and portable review packaging."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_review_docs as pack


class ReviewDocsPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.docs = self.repo / "docs"
        self.docs.mkdir(parents=True)
        self.results = self.root / "approved-results"
        self.results.mkdir()
        self.out = self.root / "bundle-v1"

    def tearDown(self):
        self.temporary.cleanup()

    def run_package(self, **kwargs):
        options = dict(repo_root=self.repo, output_dir=self.out,
                       reports=[], results_roots=[self.results])
        options.update(kwargs)
        return pack.package(**options)

    def test_category_rewrite_manifest_zip_and_sources_unchanged(self):
        goal = self.docs / "hand-object-segmentation-goal.md"
        manual = self.docs / "mano-geometry-explained.md"
        report = self.results / "trial" / "REPORT.md"
        report.parent.mkdir()
        picture = report.parent / "rgb.png"
        picture.write_bytes(b"fixture-image-bytes")
        goal.write_text(f"# Goal\n[manual](mano-geometry-explained.md#hand) [report]({report})\n"
                        "[web](https://arxiv.org/abs/example) [source](../sam3/a.py:20)\n"
                        "`[literal](../sam3/literal.py)`\n", encoding="utf-8")
        manual.write_text("# MANO\n[back](hand-object-segmentation-goal.md)\n", encoding="utf-8")
        report.write_text("# Result\n![RGB](rgb.png)\n", encoding="utf-8")
        before = {path: path.read_bytes() for path in (goal, manual, report, picture)}
        receipt = self.run_package(reports=[report])
        self.assertEqual(receipt["windows_transfer_status"], "not attempted by this packager")
        manifest = json.loads((self.out / "review/manifest.json").read_text())
        self.assertTrue(manifest["source_files_verified_unchanged"])
        copied_goal = (self.out / "review/01-goals" / goal.name).read_text()
        self.assertIn("../04-mano-memory/mano-geometry-explained.md#hand", copied_goal)
        self.assertIn("../03-results/", copied_goal)
        self.assertIn("服务器路径，未打包", copied_goal)
        self.assertIn("https://arxiv.org/abs/example", copied_goal)
        self.assertIn("`[literal](../sam3/literal.py)`", copied_goal)
        archive_bytes = (self.out / "review-docs.zip").read_bytes()
        self.assertEqual(receipt["archive_sha256"], hashlib.sha256(archive_bytes).hexdigest())
        with zipfile.ZipFile(self.out / "review-docs.zip") as archive:
            self.assertEqual(set(archive.namelist()), {entry["path"] for entry in manifest["files"]} | {"manifest.json"})
            for entry in manifest["files"]:
                data = archive.read(entry["path"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(len(data), entry["bytes"])
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)
        with self.assertRaises(FileExistsError):
            self.run_package(reports=[report])

    def test_report_requires_explicit_whitelist_and_not_recursive_scan(self):
        (self.docs / "intro.md").write_text("intro")
        (self.results / "unselected.md").write_text("secret report")
        (self.results / "weights.pt").write_bytes(b"weights")
        receipt = self.run_package()
        self.assertEqual(receipt["document_count"], 1)
        manifest = json.loads((self.out / "review/manifest.json").read_text())
        self.assertFalse(any("unselected" in entry["path"] for entry in manifest["files"]))

    def test_prioritized_report_assets_and_training_category(self):
        training = self.docs / 'distributed-training' / 'README.md'
        training.parent.mkdir()
        training.write_text('# Training\n![old](old.png)')
        (training.parent / 'old.png').write_bytes(b'old')
        report = self.results / 'README.md'
        report.write_text('# Current experiment\n![new](new.png)')
        (self.results / 'new.png').write_bytes(b'new')
        self.run_package(reports=[report], start_with=[report], max_assets=1)
        index = (self.out / 'review/START_HERE.md').read_text()
        self.assertIn('## 本次先读', index)
        self.assertIn('06-training', index)
        manifest = json.loads((self.out / 'review/manifest.json').read_text())
        self.assertIn(str(self.results / 'new.png'), [item['source'] for item in manifest['files']])
        self.assertNotIn(str(training.parent / 'old.png'), [item['source'] for item in manifest['files']])
        self.assertTrue(any(item['path'].startswith('03-results/') for item in manifest['files']))

    def test_priority_does_not_expand_report_whitelist(self):
        private = self.results / 'unselected.md'
        private.write_text('# Not selected')
        with self.assertRaisesRegex(ValueError, 'already included'):
            self.run_package(start_with=[private])

    def test_sample_review_is_data_document(self):
        self.assertEqual(pack.category(Path('egohos-sample-review-20260911.md')), '02-data')

    def test_start_index_has_one_source_grounded_description_per_markdown(self):
        (self.docs / "plan.md").write_text("# **边界对照：尚未训练**\n计划，不是结果。\n")
        (self.docs / "simple.md").write_text("本文件说明冻结参考的限制。\n下一行。\n")
        report = self.results / "OVERNIGHT_REVIEW.md"
        report.write_text("# 夜间真实状态\n部分任务仍未完成。\n")
        self.run_package(reports=[report])
        index = (self.out / "review/START_HERE.md").read_text()
        self.assertIn("— 原文主题：边界对照：尚未训练", index)
        self.assertIn("— 原文主题：本文件说明冻结参考的限制。", index)
        self.assertIn("— 原文主题：夜间真实状态", index)
        self.assertEqual(sum(line.startswith('- [') for line in index.splitlines()), 3)
        self.assertTrue(list((self.out / 'review/00-start').glob('*OVERNIGHT_REVIEW.md')))
        self.assertNotIn("实验成功", index)

    def test_index_description_ignores_fenced_fake_heading_and_bounds_length(self):
        description = pack.document_description(b'```python\n# code is not a title\n```\n# Real title\n')
        self.assertEqual(description, '原文主题：Real title')
        self.assertLessEqual(len(pack.document_description(('# '+('测'*500)).encode())), 97)

    def test_explicit_small_review_video_is_portable_but_large_video_omitted(self):
        video = self.docs / "clip.mp4"
        video.write_bytes(b"small local video fixture")
        large = self.docs / "large.mp4"
        large.write_bytes(b"x" * 201)
        (self.docs / "review.md").write_text("![clip](clip.mp4)\n[large](large.mp4)\n")
        self.run_package(max_file_bytes=200)
        text = (self.out / "review/00-start/review.md").read_text()
        self.assertIn("../figures/", text)
        self.assertIn("服务器路径，未打包", text)
        self.assertEqual(len(list((self.out / "review/figures").glob("*.mp4"))), 1)

    def test_only_explicit_root_changelog_is_included(self):
        (self.docs / "intro.md").write_text("intro")
        (self.repo / "CHANGELOG.md").write_text("changes")
        (self.repo / "private-notes.md").write_text("not requested")
        receipt = self.run_package()
        self.assertEqual(receipt["document_count"], 2)
        self.assertTrue((self.out / "review/00-start/CHANGELOG.md").read_text().endswith("changes"))
        with zipfile.ZipFile(self.out / "review-docs.zip") as archive:
            self.assertFalse(any("private-notes" in name for name in archive.namelist()))

    def test_nested_dataset_notes_stay_in_data_category_with_review_receipt(self):
        nested = self.docs / "data-audits/realsense-20260910/manual-review"
        nested.mkdir(parents=True)
        note = nested / "REVIEW.md"
        note.write_text("# Review\n[feedback](human-review.json)\n")
        receipt = nested / "human-review.json"
        receipt.write_text('{"user_verbatim":"looks good"}')
        self.assertEqual(pack.category(note), "02-data")
        self.run_package()
        copied = list((self.out / "review/02-data").glob("*-REVIEW.md"))
        self.assertEqual(len(copied), 1)
        self.assertIn("../figures/", copied[0].read_text())
        self.assertTrue(any((self.out / "review/figures").glob("*-human-review.json")))

    def test_out_of_scope_or_non_markdown_report_refused(self):
        (self.docs / "intro.md").write_text("intro")
        foreign = self.root / "foreign.md"
        foreign.write_text("not approved")
        weights = self.results / "weights.pt"
        weights.write_bytes(b"checkpoint")
        for invalid in (foreign, weights):
            with self.assertRaises(ValueError):
                self.run_package(reports=[invalid])
        self.assertFalse(self.out.exists())

    def test_conversion_audit_and_loss_reference_categories(self):
        audit = self.docs / "dexycb-conversion-trust-audit.md"
        reference = self.docs / "sam3-loss-reference.md"
        numerical = self.docs / "focal-gamma0-numerical-audit-2026-09-10.md"
        resize = self.docs / "dex-mask-resize-cpu-audit-20260910.md"
        pixels = self.docs / "data-audits/dexycb-conversion-20260910/pixel-sample-audit.json"
        pixels.parent.mkdir(parents=True)
        pixels.write_text('{"scope":"bounded pixel audit"}')
        audit.write_text("# Conversion audit\n")
        reference.write_text("# Loss reference\n")
        numerical.write_text("# Numerical audit\n")
        resize.write_text("# Dataset resize audit\n")
        self.run_package()
        self.assertTrue((self.out / "review/02-data" / audit.name).is_file())
        self.assertTrue((self.out / "review/05-meeting" / reference.name).is_file())
        self.assertTrue((self.out / "review/03-results" / numerical.name).is_file())
        self.assertTrue((self.out / "review/02-data" / resize.name).is_file())
        self.assertEqual(len(list((self.out / "review/figures").glob("*-pixel-sample-audit.json"))), 1)

    def test_symlink_document_and_symlink_output_refused(self):
        foreign = self.root / "foreign.md"
        foreign.write_text("outside")
        (self.docs / "escape.md").symlink_to(foreign)
        with self.assertRaises(ValueError):
            self.run_package()
        (self.docs / "escape.md").unlink()
        (self.docs / "intro.md").write_text("intro")
        alias = self.root / "alias"
        alias.symlink_to(self.results, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_package(output_dir=alias / "bundle")

    def test_sibling_docs_alias_maps_only_copy_links_and_preserves_source_hashes(self):
        legacy_docs = self.docs
        sibling_docs = self.root / "docs"
        self.docs.rename(sibling_docs)
        legacy_docs.symlink_to(sibling_docs, target_is_directory=True)
        self.docs = sibling_docs
        (self.repo / "scripts").mkdir()
        (self.repo / "sam3").mkdir()
        picture = self.docs / "plot.png"
        picture.write_bytes(b"unaltered scientific pixels")
        target = self.docs / "goal.md"
        target.write_text("# Goal\n")
        historical = self.docs / "intro.md"
        historical.write_text(
            f"# Introduction\n[old]({legacy_docs}/goal.md)\n![plot]({legacy_docs}/plot.png)\n"
            "[old code](../scripts/missing.py)\n[old model](../sam3/model.py)\n"
            "[new code](../repo/scripts/current.py)\n"
            f"`![literal]({legacy_docs}/plot.png)`\n", encoding="utf-8")
        report = self.results / "REPORT.md"
        report.write_text(f"# Report\n![old plot]({legacy_docs}/plot.png)\n")
        provenance = self.docs / "history.json"
        provenance.write_text(json.dumps({"source":str(legacy_docs / 'plot.png'), "sha256":"sealed"}))
        before = {path:path.read_bytes() for path in (historical, target, picture, provenance, report)}
        self.run_package(docs_root=self.docs, reports=[report])
        copied = (self.out / "review/00-start/intro.md").read_text()
        self.assertIn("../01-goals/goal.md", copied)
        self.assertIn("![plot](../figures/", copied)
        self.assertIn(f"`{self.repo}/scripts/missing.py`", copied)
        self.assertIn(f"`{self.repo}/sam3/model.py`", copied)
        self.assertIn(f"`{self.repo}/scripts/current.py`", copied)
        self.assertIn(f"`![literal]({legacy_docs}/plot.png)`", copied)
        manifest = json.loads((self.out / 'review/manifest.json').read_text())
        self.assertEqual(manifest['source_docs_root'], str(self.docs))
        self.assertEqual(manifest['asset_count'], 1)
        for entry in manifest['files']:
            if entry['source'] in {str(path) for path in before}:
                self.assertEqual(entry['source_sha256'], hashlib.sha256(before[Path(entry['source'])]).hexdigest())
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_sibling_docs_default_without_legacy_alias(self):
        self.docs.rename(self.root / "docs")
        self.docs = self.root / "docs"
        (self.docs / "intro.md").write_text("# Sibling docs\n")
        self.run_package()
        manifest = json.loads((self.out / 'review/manifest.json').read_text())
        self.assertEqual(manifest['source_docs_root'], str(self.docs))

    def test_exact_sibling_alias_is_default_but_other_aliases_refused(self):
        sibling = self.root / "docs"
        sibling.mkdir()
        self.docs.rmdir()
        self.docs.symlink_to(sibling, target_is_directory=True)
        self.assertEqual(pack.resolve_docs_root(self.repo), sibling)
        self.docs.unlink()
        self.docs.symlink_to(self.results, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directly'):
            pack.resolve_docs_root(self.repo)
        bridge = self.root / "bridge"
        bridge.symlink_to(sibling, target_is_directory=True)
        self.docs.unlink()
        self.docs.symlink_to(bridge, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directly'):
            pack.resolve_docs_root(self.repo, sibling)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            pack.resolve_docs_root(self.repo, bridge)

    def test_explicit_external_docs_cannot_remap_existing_physical_legacy_tree(self):
        external = self.root / "external-docs"
        external.mkdir()
        old = self.docs / "goal.md"
        old.write_text("# Legacy goal\n")
        (external / "goal.md").write_text("# Different goal\n")
        (external / "intro.md").write_text(f"[old]({old})\n")
        self.run_package(docs_root=external)
        copied = (self.out / "review/00-start/intro.md").read_text()
        self.assertIn("服务器路径，未打包", copied)
        self.assertNotIn("../01-goals/goal.md", copied)

    def test_alias_does_not_allow_nested_symlinks_or_unapproved_adjacent_assets(self):
        sibling = self.root / "docs"
        self.docs.rename(sibling)
        self.docs.symlink_to(sibling, target_is_directory=True)
        adjacent = self.root / "foreign.png"
        adjacent.write_bytes(b"out of scope")
        (sibling / "intro.md").write_text("![foreign](../foreign.png)\n![alias](alias.png)\n")
        (sibling / "alias.png").symlink_to(adjacent)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.run_package(docs_root=sibling)
        (sibling / "alias.png").unlink()
        self.run_package(docs_root=sibling)
        manifest = json.loads((self.out / 'review/manifest.json').read_text())
        self.assertEqual(manifest['asset_count'], 0)
        self.assertIn('outside approved roots or unavailable', {row['reason'] for row in manifest['omitted_assets']})

    def test_missing_outside_symlink_and_budget_assets_not_fake_links(self):
        image = self.results / "rgb.png"
        image.write_bytes(b"image")
        escape = self.results / "escape.png"
        escape.symlink_to(image)
        doc = self.docs / "results.md"
        doc.write_text(f"![image]({image})\n![escape]({escape})\n![missing](missing.png)\n")
        receipt = self.run_package(max_assets=0)
        self.assertEqual(receipt["asset_count"], 0)
        copied = (self.out / "review/03-results/results.md").read_text()
        self.assertNotIn("![image]", copied)
        self.assertEqual(copied.count("服务器路径，未打包"), 3)
        manifest = json.loads((self.out / "review/manifest.json").read_text())
        self.assertIn("symlink refused", {entry["reason"] for entry in manifest["omitted_assets"]})

    def test_fenced_commands_and_reference_style_links(self):
        goal = self.docs / "goal.md"
        goal.write_text("# Goal\n")
        doc = self.docs / "intro.md"
        doc.write_text("[goal][g]\n\n[g]: goal.md#section\n\n```sh\n[code](../source.py)\n```\n")
        self.run_package()
        copied = (self.out / "review/00-start/intro.md").read_text()
        self.assertIn("[g]: ../01-goals/goal.md#section", copied)
        self.assertIn("```sh\n[code](../source.py)\n```", copied)

    def test_source_mutation_prevents_publication_keeps_original_and_staging(self):
        doc = self.docs / "intro.md"
        doc.write_text("before")
        original = Path.read_bytes
        count = 0

        def changing_read(path):
            nonlocal count
            if path == doc:
                count += 1
                if count == 2:
                    doc.write_text("user edit")
            return original(path)

        with patch.object(Path, "read_bytes", changing_read):
            with self.assertRaises(RuntimeError):
                self.run_package()
        self.assertFalse(self.out.exists())
        self.assertEqual(doc.read_text(), "user edit")
        self.assertEqual(len(list(self.root.glob(".bundle-v1-building-*"))), 1)

    def test_windows_paths_traversal_and_case_collisions(self):
        for name in ("/absolute", "../a", "a/../b", "a\\b", "C:/x", "a//b", "a/NUL.png", "x:stream", "x.", "x ", "./a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pack.safe_member(name)
        (self.docs / "Goal.md").write_text("first")
        (self.docs / "goal.md").write_text("second")
        with self.assertRaises(ValueError):
            self.run_package()

    def test_long_names_are_shortened_without_losing_extension(self):
        compact = pack.compact_name("a" * 150 + ".md")
        self.assertEqual(len(compact), 80)
        self.assertTrue(compact.endswith(".md"))
        self.assertNotEqual(compact, pack.compact_name("a" * 149 + "b.md"))

    def test_size_limit_and_input_nested_output_refused(self):
        (self.docs / "intro.md").write_text("too much")
        with self.assertRaises(ValueError):
            self.run_package(max_file_bytes=2)
        with self.assertRaises(ValueError):
            self.run_package(output_dir=self.docs / "recursive")
        with self.assertRaises(ValueError):
            self.run_package(max_assets=513)

    def test_windows_script_has_only_download_not_remote_execution(self):
        script = (Path(__file__).resolve().parents[1] / "scripts/sync_review_docs_windows.ps1").read_text()
        self.assertIn("[string] $RemoteRoot", script)
        self.assertIn("[string] $DestinationRoot", script)
        self.assertNotIn('zhengyuxi', script)
        self.assertNotIn('jixiegeming', script)
        self.assertIn('StrictHostKeyChecking=yes', script)
        self.assertIn('BatchMode=yes', script)
        self.assertIn("$ExpectedSha256", script)
        self.assertIn("Assert-ArchivePath", script)
        self.assertIn("[System.IO.Directory]::Move", script)
        self.assertNotIn("Invoke-Expression", script)
        self.assertNotIn("Remove-Item", script)
        self.assertNotIn("Expand-Archive", script)
        self.assertNotIn("ssh ", script)


if __name__ == "__main__":
    unittest.main()
