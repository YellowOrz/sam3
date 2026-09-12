#!/usr/bin/env python3
"""Publish the approved DexYCB-half + nakehand training selection without copying RGB.

Only DexYCB's existing train split is sampled. All three old nakehand development
splits enter training except one explicitly excluded frame. DexYCB val remains
validation; RealSense is reserved for testing. Existing source data are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re


FORMAT = "sam3-residual-mixed-dataset-v1"
APPROVAL_FORMAT = "sam3-residual-training-selection-v1"
NAKE_SPLITS = ("train", "val", "development_holdout")
EXCLUDED_FRAME = {"source_dataset": "nakehand", "source_split": "val",
                  "recording": "nakehandego/20260907_142020", "frame": 0,
                  "source_image_id": 4713}
CATEGORIES = [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]
LABEL_LIMITATIONS = (
    "nakehand references were assisted by SAM3 prompting/propagation and are not "
    "independent human pixel ground truth; only a small prior frame subset was "
    "reviewed. Missed/incorrect hand labels remain possible. DexYCB references "
    "retain the existing rendered-label and minimum-mask-area export policy. "
    "exhaustive_hand_labels records the user's decision to use these existing "
    "references as the two-side training supervision contract, not a claim that "
    "every frame has been manually certified."
)
TEST_POLICY = "RealSense is test only; no training or LR/threshold selection"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _inside(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"Path escapes the supplied source roots: {path}")
    if not resolved.is_file():
        raise ValueError(f"Expected a source file: {path}")
    return resolved


def _document(path: Path, root: Path, documents: dict) -> tuple[dict, str]:
    path = _inside(path, (root,))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    documents[str(path)] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    return json.loads(raw), digest


def _coco_counts(coco: dict) -> dict:
    images, annotations = coco["images"], coco["annotations"]
    counts = Counter(row["image_id"] for row in annotations)
    sides = Counter(row["category_id"] for row in annotations)
    return {"images": len(images), "annotations": len(annotations),
            "left_annotations": sides[1], "right_annotations": sides[2],
            "empty_images": sum(counts[row["id"]] == 0 for row in images),
            "one_hand_images": sum(counts[row["id"]] == 1 for row in images),
            "two_hand_images": sum(counts[row["id"]] == 2 for row in images)}


def _validate_coco(coco: dict) -> dict[int, list[dict]]:
    if {row["id"]: row["name"] for row in coco["categories"]} != {1: "left_hand", 2: "right_hand"}:
        raise ValueError("Source COCO must use 1=left_hand and 2=right_hand")
    grouped, annotation_ids, sides = {}, set(), set()
    for image in coco["images"]:
        image_id = image["id"]
        if type(image_id) is not int or image_id < 0 or image_id in grouped:
            raise ValueError("Duplicate or invalid source image ID")
        if any(type(image.get(key)) is not int or image[key] <= 0 for key in ("height", "width")):
            raise ValueError("Invalid source image dimensions")
        if not isinstance(image.get("file_name"), str) or not image["file_name"]:
            raise ValueError("Missing source RGB filename")
        grouped[image_id] = []
    for annotation in coco["annotations"]:
        annotation_id, image_id, side = annotation["id"], annotation["image_id"], annotation["category_id"]
        if (type(annotation_id) is not int or annotation_id < 0 or annotation_id in annotation_ids
                or image_id not in grouped or side not in (1, 2) or (image_id, side) in sides):
            raise ValueError("Duplicate/orphan/invalid side annotation")
        annotation_ids.add(annotation_id)
        sides.add((image_id, side))
        grouped[image_id].append(annotation)
    return grouped


def _subject(image: dict) -> str:
    for key in ("subject", "subject_id", "participant_id"):
        if image.get(key) is not None:
            return str(image[key])
    sequence = str(image.get("sequence", ""))
    match = re.match(r"(subject[-_]\d+)(?:_|/|$)", sequence)
    return match.group(1) if match else "unknown_subject"


def stratified_half(images: list[dict], annotations: list[dict], seed: int = 123):
    """Hamilton allocation by existing side presence and subject, then seeded sampling.

    No model scores enter this operation. Ties use the sorted (side, subject)
    key, and both source rows and strata are sorted before the private RNG is used.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    by_side = defaultdict(set)
    for annotation in annotations:
        by_side[annotation["image_id"]].add(annotation["category_id"])
    labels = {(): "empty", (1,): "left_only", (2,): "right_only", (1, 2): "both"}
    strata = defaultdict(list)
    for image in images:
        strata[(labels[tuple(sorted(by_side[image["id"]]))], _subject(image))].append(image["id"])
    total, target = len(images), len(images) // 2
    if not total:
        return [], []
    quotas = {key: len(rows) * target // total for key, rows in strata.items()}
    remainders = {key: len(rows) * target % total for key, rows in strata.items()}
    for key in sorted(strata, key=lambda key: (-remainders[key], key))[:target - sum(quotas.values())]:
        quotas[key] += 1
    rng, selected, report = random.Random(seed), [], []
    for key in sorted(strata):
        candidates = sorted(strata[key])
        rng.shuffle(candidates)
        selected.extend(candidates[:quotas[key]])
        report.append({"side": key[0], "subject": key[1], "available": len(candidates),
                       "selected": quotas[key], "remainder_numerator": remainders[key],
                       "remainder_denominator": total})
    if len(selected) != target or len(selected) != len(set(selected)):
        raise ValueError("Half-sampling did not produce the exact unique target count")
    return sorted(selected), report


def _read_sources(dex_root: Path, nake_root: Path) -> tuple[dict, dict, dict]:
    documents, sources = {}, {}
    for split in ("train", "val"):
        coco, digest = _document(dex_root / split / "annotations.json", dex_root, documents)
        grouped = _validate_coco(coco)
        if coco.get("info", {}).get("split") != split:
            raise ValueError(f"DexYCB {split} annotations identify another split")
        if coco.get("info", {}).get("source_manifest"):
            _document(Path(coco["info"]["source_manifest"]), dex_root, documents)
        sources[("dexycb", split)] = {"root": dex_root, "coco": coco, "sha256": digest,
                                      "grouped": grouped, "rgb_hashes": {}}
    # Legacy DexYCB exports do not have READY receipts; record that explicitly.
    dex_ready = {"status": "legacy_source_without_READY", "annotations_hashed": True}
    if (dex_root / "READY.json").exists():
        ready, digest = _document(dex_root / "READY.json", dex_root, documents)
        if ready.get("status") != "complete":
            raise ValueError("DexYCB READY is incomplete")
        dex_ready = {"status": "complete", "ready_sha256": digest}
    ready, ready_hash = _document(nake_root / "READY.json", nake_root, documents)
    manifest, manifest_hash = _document(nake_root / "manifest.json", nake_root, documents)
    plan, plan_hash = _document(nake_root / "frozen-plan.json", nake_root, documents)
    if (ready.get("status") != "complete" or manifest.get("status") != "complete"
            or manifest.get("sources_unchanged") is not True
            or ready.get("manifest_sha256") != manifest_hash
            or ready.get("frozen_plan_sha256") != plan_hash
            or manifest.get("frozen_plan_sha256") != plan_hash
            or ready.get("splits") != manifest.get("splits")
            or Path(plan.get("output", "")).resolve() != nake_root):
        raise ValueError("nakehand root READY/manifest/frozen-plan binding is incomplete or changed")
    if "recovery_receipt" in ready:
        _, recovery_hash = _document(nake_root / ready["recovery_receipt"], nake_root, documents)
        if ready.get("recovery_receipt_sha256") != recovery_hash:
            raise ValueError("nakehand recovery receipt SHA mismatch")
    seen_nake_ids, seen_nake_frames, split_receipts = set(), set(), {}
    for split in NAKE_SPLITS:
        directory = nake_root / split
        receipt, receipt_hash = _document(directory / "READY.json", nake_root, documents)
        split_manifest, split_hash = _document(directory / "manifest.json", nake_root, documents)
        coco, annotation_hash = _document(directory / "annotations.json", nake_root, documents)
        grouped = _validate_coco(coco)
        counts = _coco_counts(coco)
        bound = ready["splits"].get(split, {})
        expected = {"status": "complete", "annotations_sha256": annotation_hash,
                    "manifest_sha256": split_hash, "frozen_plan_sha256": plan_hash, "counts": counts}
        if (any(receipt.get(key) != value or bound.get(key) != value for key, value in expected.items())
                or bound.get("ready_sha256") != receipt_hash
                or split_manifest.get("annotations_sha256") != annotation_hash
                or split_manifest.get("frozen_plan_sha256") != plan_hash
                or split_manifest.get("status") != "complete"
                or split_manifest.get("sources_unchanged") is not True
                or split_manifest.get("counts") != counts
                or coco.get("info", {}).get("frozen_plan_sha256") != plan_hash
                or plan.get("splits", {}).get(split, {}).get("images") != counts["images"]):
            raise ValueError(f"nakehand {split} READY/annotations/manifest/plan SHA or counts changed")
        rgb_hashes = {}
        for row in split_manifest["image_outputs"]:
            if row["image_id"] in rgb_hashes:
                raise ValueError("Duplicate nakehand RGB fingerprint")
            rgb_hashes[row["image_id"]] = row["files"]["rgb"]
        if set(rgb_hashes) != set(grouped):
            raise ValueError("nakehand RGB fingerprints do not cover all source images")
        for image in coco["images"]:
            identity = (image["recording_id"], image["frame_index"])
            if image["id"] in seen_nake_ids or identity in seen_nake_frames:
                raise ValueError("nakehand source splits overlap")
            seen_nake_ids.add(image["id"])
            seen_nake_frames.add(identity)
            if rgb_hashes[image["id"]]["path"] != image["file_name"]:
                raise ValueError("nakehand RGB fingerprint file mapping changed")
        sources[("nakehand", split)] = {"root": nake_root, "coco": coco, "sha256": annotation_hash,
                                        "grouped": grouped, "rgb_hashes": rgb_hashes}
        split_receipts[split] = {**expected, "ready_sha256": receipt_hash}
    if len(seen_nake_ids) != manifest.get("total_images") or len(seen_nake_ids) != plan.get("total_images"):
        raise ValueError("nakehand root inventory does not match all three source splits")
    return sources, documents, {"dexycb": dex_ready, "nakehand": {
        "status": "complete", "ready_sha256": ready_hash, "manifest_sha256": manifest_hash,
        "frozen_plan_sha256": plan_hash, "splits": split_receipts,
        "scope": "Published receipts, annotations, plan and selected RGB; upstream videos are not reread"}}


def _source_identity(dataset: str, split: str, image: dict) -> dict:
    return {"source_dataset": dataset, "source_split": split, "source_image_id": image["id"],
            "recording": image.get("recording_id", image.get("sequence")),
            "frame": image.get("frame_index", image.get("source_frame_index"))}


def _export_split(output: Path, split: str, entries: list, sources: dict,
                  first_image_id: int, first_annotation_id: int):
    directory = output / split
    (directory / "images").mkdir(parents=True)
    coco = {"info": {"description": "Approved residual mixed training selection", "split": split,
                     "dataset_role": split, "label_limitations": LABEL_LIMITATIONS,
                     "no_independent_nakehand_validation": True, "test_policy": TEST_POLICY},
            "categories": CATEGORIES, "images": [], "annotations": []}
    inventory, annotation_id = [], first_annotation_id
    for image_id, (dataset, old_split, original) in enumerate(entries, first_image_id):
        source = sources[(dataset, old_split)]
        rgb = _inside(source["root"] / old_split / original["file_name"], (source["root"],))
        digest = sha256(rgb)
        expected = source["rgb_hashes"].get(original["id"], {}).get("sha256")
        if expected is not None and digest != expected:
            raise ValueError(f"Source RGB SHA changed: {rgb}")
        if original.get("source_rgb_sha256") not in (None, digest):
            raise ValueError(f"Source COCO RGB SHA changed: {rgb}")
        filename = f"images/{image_id:08d}{rgb.suffix.lower()}"
        (directory / filename).symlink_to(rgb)
        identity = _source_identity(dataset, old_split, original)
        image = deepcopy(original)
        image.update(identity)
        image.update(id=image_id, file_name=filename, dataset_role=split, primary_test=False,
                     source_image_file_name=original["file_name"], source_rgb_path=str(rgb),
                     source_rgb_sha256=digest,
                     provenance={"source_root": str(source["root"]),
                                 "source_annotations_sha256": source["sha256"],
                                 "original_dataset_role": original.get("dataset_role", old_split),
                                 "original_source_rgb_path": original.get("source_rgb_path"),
                                 "original_image_provenance": original.get("provenance")})
        coco["images"].append(image)
        inventory.append({"image_id": image_id, **identity, "file_name": filename,
                          "source_rgb_path": str(rgb), "source_rgb_sha256": digest})
        for original_annotation in sorted(source["grouped"][original["id"]], key=lambda row: row["id"]):
            annotation = deepcopy(original_annotation)
            annotation.update(id=annotation_id, image_id=image_id,
                              source_annotation_id=original_annotation["id"],
                              source_dataset=dataset, source_split=old_split)
            coco["annotations"].append(annotation)
            annotation_id += 1
    atomic_json(directory / "annotations.json", coco)
    counts = _coco_counts(coco)
    manifest = {"format": FORMAT, "status": "complete", "dataset_role": split, "counts": counts,
                "annotations_sha256": sha256(directory / "annotations.json"), "image_inventory": inventory}
    atomic_json(directory / "manifest.json", manifest)
    return {"status": "complete", "annotations_sha256": manifest["annotations_sha256"],
            "manifest_sha256": sha256(directory / "manifest.json"), "counts": counts}, annotation_id


def validate_publication(output_dir: Path, *, verify_rgb: bool = True, require_ready: bool = True) -> dict:
    """Validate derived metadata, source SHA bindings and scoped RGB symlinks.

    Public callers require READY. The exporter uses require_ready=False exactly
    once before committing receipts; partial outputs otherwise fail closed.
    """
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Invalid or partial mixed publication manifest")
    approval = json.loads((output / "training-approval.json").read_text())
    if (approval.get("format") != APPROVAL_FORMAT or approval.get("approved_by") != "user"
            or sha256(output / "training-approval.json") != manifest["training_approval_sha256"]):
        raise ValueError("Training approval is not bound to the publication")
    if require_ready:
        ready = json.loads((output / "READY.json").read_text())
        if (ready.get("status") != "complete" or ready.get("manifest_sha256") != sha256(output / "manifest.json")
                or ready.get("training_approval_sha256") != manifest["training_approval_sha256"]
                or ready.get("splits") != manifest["splits"]):
            raise ValueError("Mixed publication READY hash binding changed")
    roots = tuple(Path(value).resolve(strict=True) for value in manifest["source_roots"].values())
    for document in manifest["source_documents"]:
        path = _inside(Path(document["path"]), roots)
        if path.stat().st_size != document["bytes"] or sha256(path) != document["sha256"]:
            raise ValueError(f"Source document SHA changed: {path}")
    seen_ids, seen_annotations, seen_rgb, seen_identities = set(), set(), set(), set()
    for split in ("train", "val"):
        directory, receipt = output / split, manifest["splits"][split]
        coco = json.loads((directory / "annotations.json").read_text())
        split_manifest = json.loads((directory / "manifest.json").read_text())
        _validate_coco(coco)
        if (sha256(directory / "annotations.json") != receipt["annotations_sha256"]
                or sha256(directory / "manifest.json") != receipt["manifest_sha256"]
                or split_manifest["annotations_sha256"] != receipt["annotations_sha256"]
                or _coco_counts(coco) != receipt["counts"] or split_manifest["counts"] != receipt["counts"]
                or coco["info"]["dataset_role"] != split
                or approval[split]["root"] != str(directory)
                or approval[split]["annotations_sha256"] != receipt["annotations_sha256"]
                or approval[split]["exhaustive_hand_labels"] is not True):
            raise ValueError(f"Mixed {split} manifest/annotations/approval/counts changed")
        if require_ready and json.loads((directory / "READY.json").read_text()) != receipt:
            raise ValueError(f"Mixed {split} READY binding changed")
        inventory = {row["image_id"]: row for row in split_manifest["image_inventory"]}
        if len(inventory) != len(coco["images"]):
            raise ValueError("Mixed image inventory coverage differs")
        for image in coco["images"]:
            identity = image["source_dataset"], image["source_split"], image["source_image_id"]
            if image["id"] in seen_ids or identity in seen_identities:
                raise ValueError("Source/image identity is reused across derived splits")
            if split == "val" and identity[:2] != ("dexycb", "val"):
                raise ValueError("Validation must contain only original DexYCB val")
            if split == "train" and identity[0] == "dexycb" and identity[1] != "train":
                raise ValueError("DexYCB val/test cannot enter training")
            link = directory / image["file_name"]
            if not link.is_symlink() or link.parent != directory / "images":
                raise ValueError("Derived RGB must be a symlink directly inside split/images")
            target = _inside(link, roots)
            if target in seen_rgb:
                raise ValueError("A source RGB is reused across derived images/splits")
            row = inventory[image["id"]]
            if (str(target) != image["source_rgb_path"] or row["source_rgb_path"] != str(target)
                    or row["source_rgb_sha256"] != image["source_rgb_sha256"]):
                raise ValueError("Derived source RGB provenance differs")
            if verify_rgb and sha256(target) != image["source_rgb_sha256"]:
                raise ValueError(f"Source RGB SHA changed: {target}")
            seen_ids.add(image["id"])
            seen_identities.add(identity)
            seen_rgb.add(target)
        for annotation in coco["annotations"]:
            if annotation["id"] in seen_annotations:
                raise ValueError("Derived annotation ID collides across splits")
            seen_annotations.add(annotation["id"])
    return manifest


def prepare_mixed_training(dex_root: Path, nake_root: Path, output_dir: Path, seed: int = 123,
                           dex_selection: str = 'half') -> dict:
    if dex_selection not in ('half', 'all'):
        raise ValueError('dex_selection must be half or all')
    if Path(output_dir).is_symlink():
        raise FileExistsError(f"Output must not already be a symlink: {output_dir}")
    dex_root, nake_root, output = (Path(path).resolve() for path in (dex_root, nake_root, output_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not exist: {output}")
    if (dex_root == nake_root or any(output.is_relative_to(root) or root.is_relative_to(output)
                                     for root in (dex_root, nake_root))):
        raise ValueError("Output must be separate from both immutable source roots")
    sources, documents, ready_verification = _read_sources(dex_root, nake_root)
    dex_train = sources[("dexycb", "train")]["coco"]
    selected, strata = (stratified_half(dex_train["images"], dex_train["annotations"], seed)
                        if dex_selection == 'half' else
                        (sorted(row['id'] for row in dex_train['images']), []))
    selected_ids = set(selected)
    train = [("dexycb", "train", image) for image in sorted(dex_train["images"], key=lambda row: row["id"])
             if image["id"] in selected_ids]
    excluded, nake_counts = [], {}
    for split in NAKE_SPLITS:
        images = sources[("nakehand", split)]["coco"]["images"]
        nake_counts[split] = len(images)
        for image in sorted(images, key=lambda row: row["id"]):
            identity = _source_identity("nakehand", split, image)
            if (identity["recording"], identity["frame"]) == (EXCLUDED_FRAME["recording"], EXCLUDED_FRAME["frame"]):
                if identity != EXCLUDED_FRAME:
                    raise ValueError("Excluded frame recording/frame matches but source split/image ID changed")
                excluded.append(identity)
            else:
                train.append(("nakehand", split, image))
    if excluded != [EXCLUDED_FRAME]:
        raise ValueError("Expected exactly the one declared excluded nakehand frame")
    val = [("dexycb", "val", image) for image in sorted(
        sources[("dexycb", "val")]["coco"]["images"], key=lambda row: row["id"])]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "IN_PROGRESS.json", {"status": "in_progress", "format": FORMAT})
    if dex_selection == 'all':
        # Keep the existing half-mixture validation byte-identical. Put the
        # expanded train in a disjoint ID namespace, never renumber validation.
        half_ids = set(stratified_half(dex_train['images'], dex_train['annotations'], seed)[0])
        legacy_train = [entry for entry in train if entry[0] != 'dexycb' or entry[2]['id'] in half_ids]
        val_image_start = len(legacy_train) + 1
        val_annotation_start = 1 + sum(len(sources[(d, s)]['grouped'][im['id']]) for d, s, im in legacy_train)
        train_receipt, _ = _export_split(output, 'train', train, sources, 1000000, 1000000)
    else:
        train_receipt, val_annotation_start = _export_split(output, 'train', train, sources, 1, 1)
        val_image_start = len(train) + 1
    val_receipt, _ = _export_split(output, "val", val, sources, val_image_start, val_annotation_start)
    receipts = {"train": train_receipt, "val": val_receipt}
    approval = {"format": APPROVAL_FORMAT, "approved_by": "user",
                "request": f"DexYCB train {dex_selection} + all nakehand except excluded frame; RealSense test",
                "label_limitations": LABEL_LIMITATIONS, "test_policy": TEST_POLICY}
    for split, source_roots in (("train", [dex_root, nake_root]), ("val", [dex_root])):
        approval[split] = {"root": str(output / split),
                           "annotations_sha256": receipts[split]["annotations_sha256"],
                           "exhaustive_hand_labels": True,
                           "allowed_image_roots": [str(root) for root in source_roots]}
    atomic_json(output / "training-approval.json", approval)
    manifest = {"format": FORMAT, "status": "complete", "source_roots": {
        "dexycb": str(dex_root), "nakehand": str(nake_root)},
        "source_documents": [documents[key] for key in sorted(documents)],
        "source_READY_verification": ready_verification, "splits": receipts,
        "selection": {"seed": seed, "method": ("all existing Dex train" if dex_selection == 'all' else "side+subject Hamilton largest remainder, private seeded shuffle"),
                      "fraction": {"numerator": 1, "denominator": 1 if dex_selection == 'all' else 2, "rounding": "floor"},
                      "strata": strata, "dex_selected_source_image_ids": selected,
                      "uses_model_performance": False},
        "selection_counts": {"dex_train_available": len(dex_train["images"]), "dex_train_selected": len(selected),
                             "nake_source_splits": nake_counts, "nake_available": sum(nake_counts.values()),
                             "nake_selected": sum(nake_counts.values()) - 1, "nake_excluded": 1,
                             "train_images": len(train), "dex_val_images": len(val)},
        "excluded_frames": excluded, "no_independent_nakehand_validation": True,
        "nake_split_policy": "Old train/val/development_holdout merged into train by user selection; no independent nakehand val remains",
        "label_limitations": LABEL_LIMITATIONS, "test_policy": TEST_POLICY,
        "training_approval_sha256": sha256(output / "training-approval.json")}
    atomic_json(output / "manifest.json", manifest)
    validate_publication(output, require_ready=False)
    for split in ("train", "val"):
        atomic_json(output / split / "READY.json", receipts[split])
    atomic_json(output / "IN_PROGRESS.json", {"status": "complete", "format": FORMAT})
    atomic_json(output / "READY.json", {"status": "complete", "manifest_sha256": sha256(output / "manifest.json"),
                                       "training_approval_sha256": manifest["training_approval_sha256"], "splits": receipts})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--nake-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(123,), default=123)
    parser.add_argument('--dex-selection', choices=('half', 'all'), default='half')
    args = parser.parse_args(argv)
    manifest = prepare_mixed_training(args.dex_root, args.nake_root, args.output_dir, args.seed, args.dex_selection)
    print(json.dumps({"output": str(args.output_dir.resolve()), "selection_counts": manifest["selection_counts"]}, indent=2))


if __name__ == "__main__":
    main()
