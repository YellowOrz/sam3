#!/usr/bin/env python3
"""Export honest, CPU-only residual learning curves from one TrainingMonitor run.

JSONL is preferred because it retains the original numeric values. TensorBoard
events are a fallback (float32 scalar precision). No model, CUDA, OpenCV or
Matplotlib dependency is required. Output must be a new directory outside this
repository. Loss records are sampled batches, never inferred epoch means.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOSS_KEYS = ("loss_mask", "loss_dice", "loss_bbox", "loss_giou", "loss_ce", "presence_loss")
COLORS = ("#2474b5", "#d96b27", "#33855c", "#8c56a2", "#b54560", "#4a8d99")
ROLLING_WINDOW = 20


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is not finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("global_step must be a nonnegative integer")
    return value


def _add(series, tag, step, value):
    if not isinstance(tag, str) or not tag or any(ord(char) < 32 for char in tag):
        raise ValueError("Invalid scalar tag")
    step, value = _step(step), _number(value, tag)
    values = series.setdefault(tag, {})
    if step in values and values[step] != value:
        raise ValueError(f"Conflicting duplicate scalar at {tag}, step {step}")
    duplicate = step in values
    values[step] = value
    return duplicate


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_log(log_dir, *, source="auto"):
    """Read one run only, rejecting conflicting tag/step duplicates.

    An unfinished final JSONL line is excluded explicitly for live snapshots;
    malformed complete records are errors. Images are metadata, not scalars.
    """
    path = Path(log_dir).resolve()
    if not path.is_dir():
        raise ValueError("--log-dir must be an existing tensorboard directory")
    if source not in ("auto", "jsonl", "events"):
        raise ValueError("source must be auto, jsonl or events")
    jsonl = path / "metrics.jsonl"
    selected = "jsonl" if source == "auto" and jsonl.is_file() else ("events" if source == "auto" else source)
    series, warnings, duplicates = {}, [], 0
    metadata = {"log_dir": str(path), "source_format": selected}
    if selected == "jsonl":
        raw = jsonl.read_bytes()
        metadata.update(source_file=str(jsonl), captured_bytes=len(raw),
                        captured_sha256=hashlib.sha256(raw).hexdigest())
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
            warnings.append("Excluded unfinished final JSONL line from this live snapshot.")
        previous_step, previous_samples, previous_wall = -1, -1, -1.
        for index, line in enumerate(lines, 1):
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(record, dict):
                    raise ValueError("Record must be an object")
                step = _step(record["global_step"])
                if step < previous_step:
                    raise ValueError("JSONL steps regressed: use a separate run directory after rollback")
                previous_step = step
                kind = record["kind"]
                if kind == "images":
                    continue
                if kind not in ("train", "validation") or not isinstance(record["values"], dict):
                    raise ValueError("Unknown record kind or malformed values")
                if kind == "validation":
                    scope = record["scope"]
                    if not isinstance(scope, str) or not scope or scope != scope.strip():
                        raise ValueError("Invalid validation scope")
                    prefix = f"validation/{scope}/"
                else:
                    prefix = ""
                    samples, wall = _step(record["samples_seen"]), _number(record["wall_seconds"], "wall_seconds")
                    if samples < previous_samples or wall < previous_wall or wall < 0:
                        raise ValueError("Training samples_seen/wall_seconds regressed")
                    previous_samples, previous_wall = samples, wall
                    duplicates += _add(series, "progress/samples_seen", step, samples)
                    duplicates += _add(series, "progress/wall_seconds", step, wall)
                for tag, value in record["values"].items():
                    duplicates += _add(series, prefix + tag, step, value)
            except (KeyError, TypeError, UnicodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL record {index}: {exc}") from exc
        metadata["complete_records"] = len(lines)
    else:
        files = sorted(path.glob("events.out.tfevents*"))
        if not files:
            raise ValueError("No metrics.jsonl or TensorBoard event files found")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(path), size_guidance={"scalars": 0, "images": 1,
                                                           "histograms": 1, "tensors": 1},
                                  purge_orphaned_data=False)
        events.Reload()
        for tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                duplicates += _add(series, tag, event.step, event.value)
        metadata["event_files"] = [str(file) for file in files]
        warnings.append("Event fallback uses stored scalar precision; live event snapshots are not atomic.")
    if not series:
        raise ValueError("No scalar records are available yet")
    metadata["identical_duplicate_scalars_deduplicated"] = duplicates
    return {"series": {tag: dict(sorted(points.items())) for tag, points in sorted(series.items())},
            "source": metadata, "warnings": warnings}


def rolling_mean(points, window=ROLLING_WINDOW):
    """Trailing mean of actual records only, min_periods=1, never filling steps."""
    if type(window) is not int or window < 1:
        raise ValueError("rolling window must be positive")
    queue, result = deque(), []
    for step, value in points:
        queue.append(_number(value, "rolling value"))
        if len(queue) > window:
            queue.popleft()
        result.append((step, math.fsum(queue) / len(queue)))
    return result


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _light(color):
    rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    return tuple(round(.24 * channel + .76 * 255) for channel in rgb)


def _panel(draw, box, title, curves, *, xlabel="Optimizer global step", smooth=False, validation=False,
           limits=None):
    left, top, right, bottom = box
    draw.text((left + 18, top + 10), title, font=_font(20), fill="#202733")
    available = [curve for curve in curves if curve[1]]
    if not available:
        draw.text((left + 30, top + 80), "No matching records in this snapshot", font=_font(18), fill="#657080")
        return
    legend_rows = len(available)
    x0, x1, y0, y1 = left + 95, right - 25, top + 64 + legend_rows * 25, bottom - 62
    all_points = [point for _, points, _ in available for point in points]
    xmin, xmax = min(x for x, _ in all_points), max(x for x, _ in all_points)
    ymin, ymax = limits or (min(0., min(y for _, y in all_points)), max(y for _, y in all_points))
    if xmax == xmin:
        xmin, xmax = max(0., xmin - .5), xmax + .5
    if ymax <= ymin:
        ymax = ymin + 1.
    if limits is None:
        ymax += .06 * (ymax - ymin)
    def xy(point):
        x, y = point
        return (x0 + (x - xmin) / (xmax - xmin) * (x1 - x0),
                y1 - (y - ymin) / (ymax - ymin) * (y1 - y0))
    for index in range(6):
        fraction = index / 5
        x, y = x0 + fraction * (x1 - x0), y1 - fraction * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#e1e6ed", width=1)
        draw.text((x0 - 80, y - 9), f"{ymin + fraction * (ymax-ymin):.4g}", font=_font(15), fill="#4b5563")
        draw.text((x - 22, y1 + 10), f"{xmin + fraction * (xmax-xmin):.4g}", font=_font(15), fill="#4b5563")
    draw.line((x0, y0, x0, y1, x1, y1), fill="#606b7b", width=2)
    draw.text((x0 + (x1-x0)/2 - 100, y1 + 36), xlabel, font=_font(16), fill="#374151")
    for index, (name, points, color) in enumerate(available):
        label = f"{name} | n={len(points)}"
        if smooth:
            label += f" | trailing {ROLLING_WINDOW} records, min_periods=1"
        draw.rectangle((left + 25, top + 44 + index * 25, left + 45, top + 57 + index * 25), fill=color)
        draw.text((left + 56, top + 39 + index * 25), label, font=_font(15), fill="#374151")
        coordinates = [xy(point) for point in points]
        if not validation and len(coordinates) > 1:
            draw.line(coordinates, fill=_light(color) if smooth else color, width=2)
        if smooth:
            average = [xy(point) for point in rolling_mean(points)]
            if len(average) > 1:
                draw.line(average, fill=color, width=3)
            else:
                x, y = average[0]
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        for x, y in coordinates:
            radius = 5 if validation else 2
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color if validation else _light(color))


def _figure(path, title, panels, *, columns=1, provenance=""):
    rows = (len(panels) + columns - 1) // columns
    width, height = (1540 if columns == 2 else 1440), 118 + rows * 440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), title, font=_font(26), fill="#172033")
    draw.text((30, 57), "Observed log records only; loss is sampled batch loss, not an epoch mean.",
              font=_font(17), fill="#5b6574")
    draw.text((30, 83), provenance, font=_font(13), fill="#687587")
    for index, panel in enumerate(panels):
        column, row = index % columns, index // columns
        box = (column * width // columns, 112 + row * 440,
               (column+1) * width // columns, 112 + (row+1) * 440)
        _panel(draw, box, **panel)
    image.save(path, format="PNG")


def export_plots(log_dir, output_dir, *, source="auto", validation_scope="dexycb_val"):
    """Export PNGs plus source-bound, exact numeric JSON to a fresh external directory."""
    log = load_log(log_dir, source=source)
    output = Path(output_dir).resolve()
    project = Path(__file__).resolve().parents[1]
    if output == project or project in output.parents:
        raise ValueError("Plot output must be outside the repository")
    input_dir = Path(log_dir).resolve()
    if output == input_dir or input_dir in output.parents:
        raise ValueError("Output must not modify the source log directory")
    if not isinstance(validation_scope, str) or not validation_scope or "/" in validation_scope:
        raise ValueError("validation_scope must be one nonempty scope component")
    series = log["series"]
    prefix = f"validation/{validation_scope}/"
    macro = {}
    left, right = (series.get(prefix + f"{side}/miss_zero_dice", {}) for side in ("left_hand", "right_hand"))
    for step in sorted(left.keys() & right.keys()):
        counts = [series.get(prefix + f"{side}/positive_count", {}).get(step)
                  for side in ("left_hand", "right_hand")]
        if all(count is not None and count > 0 for count in counts):
            if not 0 <= left[step] <= 1 or not 0 <= right[step] <= 1:
                raise ValueError("Validation miss-zero Dice must be in [0,1]")
            macro[step] = (left[step] + right[step]) / 2.
        else:
            log["warnings"].append(f"Macro Dice omitted at step {step}: both positive counts are required.")
    macro_tag = prefix + "derived_macro_miss_zero_dice"
    if macro_tag in series and series[macro_tag] != macro:
        raise ValueError("Logged derived macro conflicts with the observed left/right Dice and positive counts")
    if macro:
        series[macro_tag] = macro

    def curve(tag, label, color):
        return (label, list(series.get(tag, {}).items()), color)
    def panel(title, curves, **kwargs):
        return {"title": title, "curves": curves, **kwargs}
    figures = []
    spatial = curve("loss/mask_objective", "Spatial mask focal + Dice", COLORS[0])
    if spatial[1]:
        figures.append(("spatial_mask_objective.png", "Spatial objective only (not residual total loss)", [
            panel("Sampled batch focal + Dice / stage optimizer step", [spatial], smooth=True)], 1))
        spatial_components = [panel(key, [curve("loss/" + key, key, COLORS[index])], smooth=True)
                              for index, key in enumerate(LOSS_KEYS)]
        figures.append(("spatial_loss_components.png",
            "Spatial: mask/Dice optimized; box/class/presence diagnostic only", spatial_components, 2))
    total = curve("train/total_loss", "Total loss", COLORS[0])
    if total[1]:
        wall = series.get("progress/wall_seconds", {})
        time_points = [(wall[step] / 60., value) for step, value in total[1] if step in wall]
        figures.append(("training_total_loss.png", "Training total loss: faint raw + trailing 20-record mean", [
            panel("Sampled batch total loss / optimizer step", [total], smooth=True),
            panel("Same observed loss / elapsed run minutes", [("Total loss", time_points, COLORS[0])],
                  smooth=True, xlabel="Elapsed run wall minutes (logged)")], 1))
    components = [panel(key, [curve("train/" + key, key, COLORS[index])], smooth=True)
                  for index, key in enumerate(LOSS_KEYS)]
    if any(item["curves"][0][1] for item in components):
        figures.append(("training_loss_components.png", "Six original task-loss terms: sampled batches, 20-record mean", components, 2))
    dice = [curve(prefix + f"{side}/miss_zero_dice", side, COLORS[index])
            for index, side in enumerate(("left_hand", "right_hand"))]
    dice.append(curve(macro_tag, "Equal-side macro (both positive counts required)", COLORS[2]))
    candidate = [curve(prefix + f"{side}/candidate_dice", side, COLORS[index])
                 for index, side in enumerate(("left_hand", "right_hand"))]
    if any(item[1] for item in dice + candidate):
        figures.append(("validation_dice.png", f"Validation [{validation_scope}]: measured points only; no interpolation", [
            panel("Positive-sample Dice, missed detection counts as zero", dice, validation=True, limits=(0., 1.)),
            panel("Candidate Dice (does not penalize missed detection)", candidate, validation=True, limits=(0., 1.))], 1))
    boundaries = [panel(f"{kind} Boundary IoU, fixed 4 original-image pixels", [
        curve(prefix + f"{side}/{kind}_boundary_iou_4px", side, COLORS[index])
        for index, side in enumerate(("left_hand", "right_hand"))], validation=True, limits=(0., 1.))
        for kind in ("miss_zero", "candidate")]
    if any(curve[1] for item in boundaries for curve in item["curves"]):
        figures.append(("validation_boundary.png", f"Validation [{validation_scope}] boundary: measured points only", boundaries, 1))
    detections = [panel(name.replace("_", " "), [
        curve(prefix + f"{side}/{name}", side, COLORS[index])
        for index, side in enumerate(("left_hand", "right_hand"))], validation=True,
        limits=(0., 1.) if name.endswith("rate") else None)
        for name in ("false_negative_rate", "false_positive_rate", "false_negative_count", "false_positive_count")]
    if any(curve[1] for item in detections for curve in item["curves"]):
        figures.append(("validation_detection.png", f"Validation [{validation_scope}] FN / FP: measured points only", detections, 2))
    timing = [panel("Measured throughput / log window", [curve("perf/images_per_second", "Images / second", COLORS[0])], smooth=True),
              panel("Measured stage times / optimizer step (ms)", [curve("perf/" + key, label, COLORS[index])
                    for index, (key, label) in enumerate((("data_wait_ms_per_step", "Data wait"),
                                                         ("h2d_ms_per_step", "Host to device"),
                                                         ("compute_ms_per_step", "Compute")))], smooth=True),
              panel("Elapsed run wall time (seconds)", [curve("progress/wall_seconds", "Run seconds", COLORS[0])]),
              panel("Observed cumulative training samples", [curve("progress/samples_seen", "Samples seen", COLORS[1])])]
    if any(curve[1] for item in timing for curve in item["curves"]):
        figures.append(("throughput_timing.png", "Performance: observed log windows and counters", timing, 2))
    if not figures:
        raise ValueError("No matching training/performance/validation curve tags found")
    for _, _, panels, _ in figures:
        for item in panels:
            limits = item.get("limits")
            if limits is not None and any(not limits[0] <= y <= limits[1]
                                          for _, points, _ in item["curves"] for _, y in points):
                raise ValueError(f"Metric outside its declared plotting range: {item['title']}")
    output.mkdir(parents=True, exist_ok=False)
    provenance = (f"Source: {input_dir.parent.name}/{input_dir.name} | {log['source']['source_format']} | "
                  f"captured SHA256: {log['source'].get('captured_sha256', 'event snapshot')[:16]}")
    for name, title, panels, columns in figures:
        _figure(output / name, title, panels, columns=columns, provenance=provenance)
    summary = {
        "schema": "sam3-residual-training-plots-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": log["source"], "warnings": log["warnings"], "validation_scope": validation_scope,
        "loss_interpretation": "Logged current batch loss sampled at logging steps; NOT epoch averages.",
        "spatial_interpretation": "loss/mask_objective is focal + Dice only; not comparable to residual total loss. Other spatial loss terms are diagnostics.",
        "smoothing": {"window_records": ROLLING_WINDOW, "min_periods": 1, "alignment": "trailing",
                      "short_series": "Uses only available records, even when fewer than 20.",
                      "raw_line": "faint", "mean_line": "solid"},
        "validation_interpolation": False,
        "validation_interpretation": "Only actual validation steps; may include baseline/partial-run diagnostics, not necessarily epoch ends.",
        "macro_definition": "Equal mean of left/right miss_zero_dice at the same step, both positive counts > 0.",
        "checkpoint_selection": "These plots do not select checkpoints; diagnostic points need not be selector decisions.",
        "files": [str(output / name) for name, *_ in figures],
        "observations": {
            "training_logged_steps": list(series.get("train/total_loss", {})),
            "validation_macro_logged_steps": list(macro),
            "last_recorded_scalars": {tag: {"step": next(reversed(points)),
                                              "value": points[next(reversed(points))]}
                                      for tag, points in sorted(series.items()) if points},
            "warning": "Latest logged values are observations, not epoch means or a checkpoint recommendation.",
        },
        "series": {tag: {"steps": list(points), "values": list(points.values()), "records": len(points)}
                   for tag, points in sorted(series.items())},
    }
    (output / "summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, action="append", required=True,
                        help="One run's tensorboard directory; do not mix unrelated runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New external directory; existing outputs are never overwritten")
    parser.add_argument("--source", choices=("auto", "jsonl", "events"), default="auto")
    parser.add_argument("--validation-scope", default="dexycb_val")
    args = parser.parse_args(argv)
    if len(args.log_dir) != 1:
        parser.error("This version accepts exactly one log directory; continuation merging is not supported")
    report = export_plots(args.log_dir[0], args.output_dir, source=args.source, validation_scope=args.validation_scope)
    print(json.dumps({"summary": str(args.output_dir.resolve() / "summary.json"), "files": report["files"]}, indent=2))


if __name__ == "__main__":
    main()
