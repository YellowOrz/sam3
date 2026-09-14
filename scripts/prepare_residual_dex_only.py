"""Publish explicitly approved Dex-only train from an audited mixed publication.

Keep image/annotation rows intact, verify RGB bytes, and retain original Dex val.
No new labels, decoding, downloads, test access, or changes to source files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.residual_ddp_data import load_coco_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_dex(data, *, split, expected_count):
    selected = [row for row in data["images"] if row.get("source_dataset") == "dexycb"]
    if len(selected) != expected_count or len({row["id"] for row in selected}) != expected_count:
        raise ValueError("Unexpected Dex coverage")
    for row in selected:
        if row.get("source_split") != split or row.get("dataset_role", split) != split:
            raise ValueError("Source Dex split disagrees with selected role")
        name = Path(row["file_name"])
        if name.is_absolute() or ".." in name.parts or name.parts[:1] != ("images",):
            raise ValueError("Source RGB filename escapes images")
    identities = {row["id"] for row in selected}
    result = deepcopy(data)
    result["images"] = deepcopy(selected)
    result["annotations"] = [deepcopy(row) for row in data["annotations"] if row["image_id"] in identities]
    result["info"] = {"dataset_role": split, "split": split,
        "description": "Dex-only subset; image/annotation rows unchanged",
        "label_limitations": data.get("info", {}).get("label_limitations", "Existing Dex reference policy retained")}
    return result


def prepare(source, output, *, expected_train=23265, expected_val=2909):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if (output.exists() or output.is_relative_to(repo) or output.is_relative_to(source)
            or source.is_relative_to(output)):
        raise ValueError("Require a new external output separate from source and repository")
    original_approval = json.loads((source / "training-approval.json").read_bytes())
    if (original_approval.get("format") != "sam3-residual-training-selection-v1"
            or original_approval.get("approved_by") != "user"):
        raise ValueError("Source must have an approved selection")
    documents, paths = {}, {}
    for split, count in (("train", expected_train), ("val", expected_val)):
        root = source / split
        declaration = original_approval[split]
        if sha(root / "annotations.json") != declaration["annotations_sha256"]:
            raise ValueError("Source annotations no longer match approved SHA")
        allowed = [source, *declaration.get("allowed_image_roots", [])]
        contract = load_coco_contract(root, approved_exhaustive=declaration.get("exhaustive_hand_labels") is True,
                                      allowed_image_roots=allowed)
        data = json.loads((root / "annotations.json").read_bytes())
        documents[split] = select_dex(data, split=split, expected_count=count)
        if split == "val" and len(data["images"]) != count:
            raise ValueError("Validation must remain entirely Dex and unchanged")
        paths[split] = set()
        for row in documents[split]["images"]:
            rgb = (root / row["file_name"]).resolve(strict=True)
            if sha(rgb) != row.get("source_rgb_sha256"):
                raise ValueError("Source RGB hash differs")
            paths[split].add(rgb)
        if len(paths[split]) != count:
            raise ValueError("Duplicate RGB path")
    if paths["train"] & paths["val"]:
        raise ValueError("Train/val physical RGB overlap")
    output.mkdir(parents=True, exist_ok=False)
    (output / "train").mkdir()
    (output / "train/images").symlink_to(source / "train/images", target_is_directory=True)
    (output / "val").symlink_to(source / "val", target_is_directory=True)
    def write(path, value):
        with path.open("x") as stream:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    write(output / "train/annotations.json", documents["train"])
    approval = {"format": "sam3-residual-training-selection-v1", "approved_by": "user",
        "request": "New experiments: Dex train only, fixed Dex val; nakehand and RealSense external evaluation",
        "source_approval_sha256": sha(source / "training-approval.json"),
        "label_limitations": original_approval.get("label_limitations"),
        "test_policy": "No nakehand/RealSense for training, validation selection or threshold tuning"}
    for split in ("train", "val"):
        approval[split] = {"root": str(output / split), "allowed_image_roots": [str(source),
            *original_approval[split].get("allowed_image_roots", [])],
            "exhaustive_hand_labels": True, "annotations_sha256": sha(output / split / "annotations.json")}
        load_coco_contract(output / split, approved_exhaustive=True,
                           allowed_image_roots=approval[split]["allowed_image_roots"])
    write(output / "training-approval.json", approval)
    result = {"status": "complete", "format": "sam3-dex-only-publication-v1", "source": str(source),
        "counts": {split: {"images": len(documents[split]["images"]),
            "annotations": len(documents[split]["annotations"])} for split in ("train", "val")},
        "all_selected_rgb_hashes_verified": True, "original_rows_unchanged": True,
        "validation_bytes_unchanged": sha(output / "val/annotations.json") == original_approval["val"]["annotations_sha256"],
        "source_annotations_sha256": {split: original_approval[split]["annotations_sha256"] for split in ("train", "val")},
        "approval_sha256": sha(output / "training-approval.json")}
    write(output / "READY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-dex-only", action="store_true", required=True,
                        help="Caller confirms explicit user authorization for this new data policy")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
