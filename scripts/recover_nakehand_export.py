#!/usr/bin/env python3
"""Recover only the interrupted nakehand development holdout, on CPU.

Published train/val are read-only. Every reused partial PNG is compared with a
fresh source-video decode before copying; unusable partial files are preserved.
The old holdout is renamed to an explicit backup only after the new split passes
the original full validation. No source file or existing PNG is overwritten.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

try:
    from scripts import prepare_nakehand_training as exporter
except ModuleNotFoundError:
    import prepare_nakehand_training as exporter


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_plan(plan_path, expected_sha):
    plan_path = Path(plan_path).resolve()
    if exporter.sha256(plan_path) != expected_sha:
        raise ValueError("Frozen plan hash differs from explicitly expected SHA256")
    output = plan_path.parent
    plan = read_json(plan_path)
    if plan["format"] != "nakehand-development-split-plan-v1" or Path(plan["output"]).resolve() != output:
        raise ValueError("Frozen plan schema/location mismatch")
    expected = exporter.split_plan({name: row["frame_count"] for name, row in plan["recordings"].items()})
    if any(plan[key] != expected[key] for key in ("recordings", "splits", "total_images")):
        raise ValueError("Frozen split allocation differs from declared protocol")
    root = Path(plan["root"]).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Source and output roots must be separate")
    if len(plan["sources"]) != 60 or len({row["path"] for row in plan["sources"]}) != 60:
        raise ValueError("Expected the frozen 60-file source inventory")
    implementation = plan["implementation_sources"]
    if {Path(row["path"]).name for row in implementation} != set(exporter.HELPERS) or len(implementation) != 3:
        raise ValueError("Frozen exporter implementation inventory mismatch")
    exporter.verify_sources(implementation)
    for row in implementation:
        snapshot = output / "implementation-snapshot" / Path(row["path"]).name
        current = Path(exporter.__file__).resolve().with_name(snapshot.name)
        if exporter.sha256(snapshot) != row["sha256"] or exporter.sha256(current) != row["sha256"]:
            raise ValueError("Frozen/current exporter implementation differs")
    exporter.verify_sources(plan["sources"])
    exporter.verify_sources([plan["prior_exposure"]["source"]])
    return output, plan


def validate_published(output, split, plan, plan_sha):
    """Full published PNG/RLE recheck, NOT a fresh train/val video decode."""
    directory = Path(output) / split
    paths = [directory / name for name in ("READY.json", "manifest.json", "annotations.json")]
    if directory.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Published split/metadata must not be symlinks")
    before = {str(path): exporter.sha256(path) for path in paths}
    ready, manifest, coco = map(read_json, paths)
    if ready["status"] != "complete" or manifest["status"] != "complete" or not manifest["sources_unchanged"]:
        raise ValueError("Published split is not complete")
    if ready["manifest_sha256"] != before[str(paths[1])] or ready["annotations_sha256"] != before[str(paths[2])]:
        raise ValueError("Published READY digest mismatch")
    if ready["frozen_plan_sha256"] != plan_sha or manifest["sources"] != plan["sources"]:
        raise ValueError("Published split plan/source identity mismatch")
    if manifest["split"] != split or coco["info"]["split"] != exporter.COCO_SPLITS[split]:
        raise ValueError("Published split role mismatch")
    expected_ids = {plan["recordings"][name]["global_image_id_offset"] + index: (name, index)
                    for name in plan["splits"][split]["recordings"]
                    for index in range(plan["recordings"][name]["frame_count"])}
    actual_ids = {image["id"]: (image["recording_id"], image["frame_index"]) for image in coco["images"]}
    if actual_ids != expected_ids or len(coco["images"]) != len(expected_ids):
        raise ValueError("Published frame allocation differs from frozen plan")
    for row in manifest["image_outputs"]:
        for item in row["files"].values():
            path = directory / item["path"]
            if Path(item["path"]).is_absolute() or directory.resolve() not in path.resolve().parents or path.is_symlink():
                raise ValueError("Unsafe published PNG path")
    validation = exporter.validate_export(directory, plan_sha)
    per_image = Counter(row["image_id"] for row in coco["annotations"])
    sides = Counter(row["category_id"] for row in coco["annotations"])
    counts = {"images": len(expected_ids), "annotations": len(coco["annotations"]),
              "left_annotations": sides[1], "right_annotations": sides[2],
              "empty_images": sum(per_image[index] == 0 for index in expected_ids),
              "one_hand_images": sum(per_image[index] == 1 for index in expected_ids),
              "two_hand_images": sum(per_image[index] == 2 for index in expected_ids)}
    if counts != ready["counts"] or counts != manifest["counts"]:
        raise ValueError("Published count summaries do not match annotations")
    if before != {str(path): exporter.sha256(path) for path in paths}:
        raise ValueError("Published split metadata changed during validation")
    return {**ready, "ready_sha256": before[str(paths[0])], "directory": split}, {
        **validation, "fresh_source_video_decode": False,
        "scope": "All current PNG hashes/pixels and RLE/area/bbox vs immutable published manifest; original exporter source-decoded every frame"}


class ReusingPNGWriter:
    """Only copy a partial PNG after exact fresh-source decoded pixel equality."""

    def __init__(self, partial, staging, original_writer):
        self.partial, self.staging = Path(partial), Path(staging)
        self.original_writer = original_writer
        self.reused, self.new, self.unusable = [], 0, []

    def __call__(self, path, array, compress_level):
        path = Path(path)
        relative = path.relative_to(self.staging)
        source = self.partial / relative
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite staged PNG: {path}")
        if source.exists() or source.is_symlink():
            try:
                if source.is_symlink() or self.partial.resolve() not in source.resolve().parents:
                    raise ValueError("Unsafe partial PNG path")
                digest = exporter.sha256(source)
                with Image.open(source) as saved:
                    if saved.format != "PNG" or not np.array_equal(np.asarray(saved), array):
                        raise ValueError("Partial PNG pixels differ from freshly decoded original source")
                if exporter.sha256(source) != digest:
                    raise ValueError("Partial PNG changed while reading")
            except (OSError, ValueError) as error:
                self.unusable.append({"path": relative.as_posix(), "reason": str(error)})
            else:
                # Copy, never hardlink: modifying a later backup cannot change the published data.
                shutil.copyfile(source, path)
                if exporter.sha256(path) != digest or exporter.sha256(source) != digest:
                    raise ValueError("Partial PNG changed while copying")
                self.reused.append({"path": relative.as_posix(), "sha256": digest})
                return {"path": path.name, "sha256": digest}
        self.new += 1
        return self.original_writer(path, array, compress_level)


def recover(plan_path, expected_sha):
    output, plan = checked_plan(plan_path, expected_sha)
    if (output / "READY.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("Root already published; recovery will not overwrite it")
    partial = output / "development_holdout"
    if not partial.is_dir() or partial.is_symlink() or (partial / "READY.json").exists():
        raise ValueError("Expected an unpublished non-symlink partial development_holdout")
    # An advisory lock only protects concurrent recovery invocations, never kills other work.
    with (output / ".recovery.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _recover_locked(Path(plan_path).resolve(), expected_sha, output, plan, partial)


def _recover_locked(plan_path, plan_sha, output, plan, partial):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = Path(tempfile.mkdtemp(prefix=f"recovery-{stamp}-", dir=output))
    backup = output / f"development_holdout.interrupted-backup-{work.name}"
    if backup.exists():
        raise FileExistsError("Backup destination already exists")
    recovery = {"format": "nakehand-export-recovery-v1", "status": "validating_published_splits",
                "frozen_plan_sha256": plan_sha, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "implementation": exporter.source_record(Path(__file__).resolve()),
                "work_directory": str(work), "partial_backup": str(backup),
                "policy": "Read-only published train/val; fresh-source pixel verified partial copies; preserve every old partial file"}
    receipt_path = work / "recovery.json"
    exporter.atomic_json(receipt_path, recovery)
    receipts, validations = {}, {}
    try:
        for split in ("train", "val"):
            print(f"Full read-only validation: {split}", flush=True)
            receipts[split], validations[split] = validate_published(output, split, plan, plan_sha)
        root = Path(plan["root"])
        recordings = {name: exporter.load_recording(root, root / name / "metadata.json") for name in plan["recordings"]}
        for recording in recordings.values():
            exporter.recording_identity(recording)
        sources = [row for recording in recordings.values() for row in recording["sources"]]
        if sources != plan["sources"]:
            raise ValueError("Fresh source inventory differs from frozen plan")
        recovery.update(status="exporting_holdout", published_validation=validations)
        exporter.atomic_json(receipt_path, recovery)
        original_writer = exporter.save_png
        writer = ReusingPNGWriter(partial, work / "development_holdout", original_writer)
        try:
            exporter.save_png = writer
            receipts["development_holdout"] = exporter.export_split(work, "development_holdout", plan, plan_sha, recordings)
        finally:
            exporter.save_png = original_writer
        checked_plan(plan_path, plan_sha)
        exporter.verify_sources([recovery["implementation"]])
        # Recheck identities after the lengthy source decode without rewriting train/val.
        for split in ("train", "val"):
            if exporter.sha256(output / split / "READY.json") != receipts[split]["ready_sha256"]:
                raise ValueError("Published READY changed during recovery")
            if exporter.sha256(output / split / "manifest.json") != receipts[split]["manifest_sha256"]:
                raise ValueError("Published manifest changed during recovery")
            if exporter.sha256(output / split / "annotations.json") != receipts[split]["annotations_sha256"]:
                raise ValueError("Published annotations changed during recovery")
        for row in writer.reused:
            if exporter.sha256(partial / row["path"]) != row["sha256"]:
                raise ValueError("Reused partial file changed before backup publication")
        recovery.update(status="ready_to_publish", reused_pngs=len(writer.reused), newly_encoded_pngs=writer.new,
                        unusable_partial_pngs=writer.unusable, reused_files=writer.reused,
                        holdout_fresh_source_decode_all_frames=True)
        exporter.atomic_json(receipt_path, recovery)
        if (output / "READY.json").exists() or (output / "manifest.json").exists() or backup.exists():
            raise FileExistsError("Publication target appeared during recovery")
        # Renames preserve the entire interrupted output, including incomplete/corrupt files.
        partial.rename(backup)
        try:
            (work / "development_holdout").rename(partial)
        except BaseException:
            if not partial.exists():
                backup.rename(partial)
            raise
        manifest = {"format": "nakehand-development-dataset-v1", "status": "complete", "frozen_plan_sha256": plan_sha,
                    "source": plan["root"], "sources": plan["sources"], "sources_unchanged": True,
                    "splits": receipts, "total_images": plan["total_images"], "label_source": exporter.LABEL_SOURCE,
                    "person_session_camera_relationship": plan["person_session_camera_relationship"],
                    "prior_exposure": plan["prior_exposure"], "development_holdout_policy": plan["development_holdout_policy"],
                    "recovery_receipt": str(receipt_path), "partial_backup": str(backup),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        exporter.atomic_json(output / "manifest.json", manifest)
        recovery.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
        exporter.atomic_json(receipt_path, recovery)
        exporter.atomic_json(output / "READY.json", {"status": "complete", "frozen_plan_sha256": plan_sha,
                            "manifest_sha256": exporter.sha256(output / "manifest.json"), "splits": receipts,
                            "recovery_receipt": str(receipt_path), "recovery_receipt_sha256": exporter.sha256(receipt_path)})
        # Keep the old in-progress marker as historical evidence; root READY is authoritative.
        print(json.dumps({"status": "complete", "output": str(output), "partial_backup": str(backup),
                          "reused_pngs": len(writer.reused), "new_pngs": writer.new}, indent=2), flush=True)
        return manifest
    except BaseException as error:
        recovery.update(status="failed", error=f"{type(error).__name__}: {error}",
                        note="No automatic deletion; work and original partial/backup are retained for inspection")
        exporter.atomic_json(receipt_path, recovery)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    recover(args.plan, args.expected_plan_sha256)


if __name__ == "__main__":
    main()
