#!/usr/bin/env python3
"""Build side-aware DexYCB manifests from the unified dataset.

The current unified DexYCB export contains a legacy conversion bug: hand masks
are correct, but ``instances.json`` hard-codes ``hand_right``.  DexYCB's
sequence-level ``extra.mano_sides`` is the authoritative side annotation, as
used by the official DexYCB toolkit.  This builder records that side explicitly
without modifying the shared unified dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VALID_SIDES = ("left", "right")
VALID_SPLITS = ("train", "val", "test")


def _load_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def get_sequence_side(sequence: Dict, sequence_path: Path) -> str:
    """Return the one authoritative DexYCB MANO side for a sequence."""
    sides = sequence.get("extra", {}).get("mano_sides")
    if not isinstance(sides, list) or len(sides) != 1 or sides[0] not in VALID_SIDES:
        raise ValueError(f"Expected exactly one left/right mano_sides in {sequence_path}")
    return sides[0]


def expand_frame_map(record: Dict, num_frames: int) -> List[int]:
    """Expand and validate the closed ranges in one instance frame map."""
    visible: List[int] = []
    for entry in record.get("frame_map", []):
        frames = entry.get("frames")
        instance_id = entry.get("id")
        if (
            not isinstance(frames, list)
            or len(frames) != 2
            or not all(isinstance(value, int) for value in frames)
            or frames[0] < 0
            or frames[0] > frames[1]
            or frames[1] >= num_frames
            or not isinstance(instance_id, int)
            or instance_id <= 0
        ):
            raise ValueError(f"Invalid hand frame_map entry: {entry!r}")
        visible.extend(range(frames[0], frames[1] + 1))
    if len(visible) != len(set(visible)):
        raise ValueError(f"Overlapping hand frame_map ranges for {record.get('track_id')}")
    return sorted(visible)


def build_manifests(unified_root: Path) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Build train/val/test view manifests and a list of hand-empty views."""
    unified_root = Path(unified_root)
    split_path = unified_root / "metadata" / "split.json"
    split_data = _load_json(split_path)
    try:
        subject_splits = split_data["dexycb"]["by_subject"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing dexycb.by_subject in {split_path}") from error

    manifests = {split: [] for split in VALID_SPLITS}
    excluded: List[Dict] = []
    sequence_paths = sorted(
        (unified_root / "sequences" / "dexycb").glob("*/sequence.json")
    )
    if not sequence_paths:
        raise FileNotFoundError("No DexYCB sequence.json files found")

    for sequence_path in sequence_paths:
        sequence = _load_json(sequence_path)
        sequence_id = sequence["seq_id"]
        subject = sequence["subject"]
        side = get_sequence_side(sequence, sequence_path)
        num_frames = int(sequence["num_frames"])
        try:
            split = subject_splits[subject]
        except KeyError as error:
            raise ValueError(f"No split for subject {subject!r}") from error
        if split not in manifests:
            raise ValueError(f"Unsupported split {split!r} for {subject}")

        for view in sequence["views"]:
            view = str(view)
            view_dir = sequence_path.parent / view
            instances_path = view_dir / "instances.json"
            instances = _load_json(instances_path).get("instances")
            if not isinstance(instances, list):
                raise ValueError(f"Missing instances list in {instances_path}")
            hands = [
                record
                for record in instances
                if str(record.get("kind", "")).startswith("hand_")
            ]
            if not hands:
                excluded.append(
                    {
                        "source": "dexycb",
                        "sequence": sequence_id,
                        "subject": subject,
                        "split": split,
                        "view": view,
                        "reason": "no_hand_instance",
                    }
                )
                hand = None
                visible_frames = []
                source_kind = None
            elif len(hands) != 1:
                raise ValueError(f"Expected one hand instance in {instances_path}, got {len(hands)}")
            else:
                hand = hands[0]
                visible_frames = expand_frame_map(hand, num_frames)
                source_kind = hand["kind"]

            expected_kind = f"hand_{side}"
            if source_kind is not None and source_kind not in (
                expected_kind,
                "hand_right",
            ):
                raise ValueError(
                    f"Unexpected hand kind {source_kind!r} for authoritative side "
                    f"{side!r} in {instances_path}"
                )

            manifests[split].append(
                {
                    "source": "dexycb",
                    "sequence": sequence_id,
                    "subject": subject,
                    "split": split,
                    "view": view,
                    "view_dir": str(view_dir),
                    "num_frames": num_frames,
                    "hand_side": side,
                    "hand_kind": expected_kind,
                    "source_hand_kind": source_kind,
                    "hand_visible_frames": visible_frames,
                    "hand_visible_frame_count": len(visible_frames),
                    "rgb_path": str(view_dir / "rgb.mkv"),
                    "mask_path": str(view_dir / "mask.mkv"),
                    "instances_path": str(instances_path),
                    "sequence_path": str(sequence_path),
                }
            )

    return manifests, excluded


def write_manifests(
    manifests: Dict[str, List[Dict]], excluded: Iterable[Dict], output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in VALID_SPLITS:
        path = output_dir / f"{split}_views.json"
        path.write_text(
            json.dumps(manifests[split], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "excluded_views.json").write_text(
        json.dumps(list(excluded), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests, excluded = build_manifests(args.unified_root)
    write_manifests(manifests, excluded, args.output_dir)
    summary = {
        split: {
            "views": len(records),
            "frames": sum(record["num_frames"] for record in records),
            "visible": sum(record["hand_visible_frame_count"] for record in records),
            "left_visible": sum(
                record["hand_visible_frame_count"]
                for record in records
                if record["hand_side"] == "left"
            ),
            "right_visible": sum(
                record["hand_visible_frame_count"]
                for record in records
                if record["hand_side"] == "right"
            ),
        }
        for split, records in manifests.items()
    }
    summary["excluded_views"] = len(excluded)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
