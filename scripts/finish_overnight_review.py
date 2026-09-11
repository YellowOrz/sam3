"""CPU-only bounded status reader and portable morning-document publisher.

Never launches training/evaluation, edits old reports, contacts Windows, signals
another job, or interprets commands stored in run state. Unknown report contracts
are reported as unverified rather than accepted as completed experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

if __package__:
    from .package_review_docs import resolve_docs_root
else:
    from package_review_docs import resolve_docs_root

FORMAT = "sam3-overnight-review-finalizer-v1"
TERMINAL = {"complete", "completed", "failed", "failed_or_stopped", "cancelled", "aborted"}
SUCCESS = {"complete", "completed"}
KNOWN_REPORTS = {
    "sam3-hand-boundary-validation-v1": ("baseline", "CP0", "CP1"),
    "sam3-input-ve-boundary-validation-v1": ("baseline", "output-delta", "input-ve"),
}
KNOWN_QUEUES = {
    "sam3-nakehand-boundary-supervisor-v1": {
        "boundary-comparison": "sam3-hand-boundary-validation-v1",
        "semantic-comparison": "sam3-hand-boundary-validation-v1",
    },
    "sam3-nakehand-input-ablation-supervisor-v1": {
        "lr1e-3": "sam3-input-ve-boundary-validation-v1",
        "lr3e-4": "sam3-input-ve-boundary-validation-v1",
    },
}
QUEUE_TRAINING_COMPARISONS = {
    "sam3-nakehand-boundary-supervisor-v1": {
        "only_training_factor": "boundary_weight: 0 versus 4", "actual_steps_each": 2000,
        "same_initial_cache": True, "same_ordered_training_images": True,
    },
    "sam3-nakehand-input-ablation-supervisor-v1": {
        "only_training_factor": "learning_rate: .001 versus .0003", "actual_steps_each": 2000,
        "same_initial_input_residual": True, "same_initial_cache": True,
        "same_ordered_training_images": True,
    },
}
SIDES = ("left_hand", "right_hand")
MAX_JSON_BYTES = 128 * 1024**2
POLL_SECONDS = 30
PUBLICATION_RESERVE_SECONDS = 120


def clean_path(value):
    path = Path(os.path.abspath(value))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Refuse symlink path: {path}")
    return path


def fingerprint(path):
    digest = hashlib.sha256()
    with clean_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = clean_path(path)
    before = path.stat()
    if not path.is_file() or before.st_size > MAX_JSON_BYTES:
        raise ValueError("JSON must be a bounded regular file")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("JSON changed while being read")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def inside_file(value, root, suffix=None):
    path = clean_path(value)
    if not path.is_relative_to(root) or not path.is_file() or (suffix and path.suffix != suffix):
        raise ValueError(f"Report artifact must be inside explicit run: {path}")
    return path


def checked_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected explicit SHA-256")
    return value


def snapshot_run(root):
    try:
        state, digest = read_json(root / "state.json")
        if not isinstance(state, dict) or not isinstance(state.get("status"), str):
            raise ValueError("State has no explicit string status")
        return {"root": str(root), "status": state["status"], "state_sha256": digest, "state": state}
    except (OSError, ValueError, RuntimeError) as error:
        return {"root": str(root), "status": "unreadable_or_not_started", "error": str(error)[:500], "state": {}}


def selected_visuals(summary, summary_path, root, labels):
    """Fixed first/middle/last of previously scheduled renders, never GT ranking."""
    visuals = summary.get('visuals', [])
    if not isinstance(visuals, list) or len(visuals) > 100:
        raise ValueError('Unbounded or malformed explicit visual inventory')
    if not visuals:
        return [], ['Final summary contains no pre-rendered visual inventory']
    if any(not isinstance(item, dict) or type(item.get('dataset_index')) is not int for item in visuals):
        raise ValueError('Visual inventory has no explicit dataset indices')
    ordered = sorted(visuals, key=lambda item:item['dataset_index'])
    if len({item['dataset_index'] for item in ordered}) != len(ordered):
        raise ValueError('Duplicate rendered dataset index')
    chosen = [ordered[index] for index in sorted({0, len(ordered)//2, len(ordered)-1})]
    identities = dict(zip(summary.get('evaluated_dataset_indices', []), summary['evaluated_image_ids']))
    result, omissions = [], []
    for visual in chosen:
        if identities.get(visual['dataset_index']) != visual.get('image_id'):
            raise ValueError('Rendered identity differs from completed evaluation inventory')
        directory = clean_path(visual['directory'])
        if directory.parent != summary_path.parent/'visuals' or not directory.is_relative_to(root):
            raise ValueError('Visual directory escaped this report')
        if clean_path(visual['comparison']) != directory/'comparison.png':
            raise ValueError('Comparison path differs from explicit rendered directory')
        candidates = [('分离对比总图', directory/'comparison.png'), ('原图 RGB', directory/'rgb.png'),
                      ('左参考', directory/'left_hand__reference.png'), ('右参考', directory/'right_hand__reference.png')]
        candidates += [(f'{label} {side} 达阈预测', directory/label/f'{side}__detected.png')
                       for label in labels for side in SIDES]
        assets = []
        for title, path in candidates:
            try:
                inside_file(path, root, '.png')
                if not 0 < path.stat().st_size <= 8*1024**2:
                    raise ValueError('Image empty or over the 8 MiB copy cap')
                assets.append({'title':title, 'path':str(path), 'sha256':fingerprint(path)})
            except (OSError, ValueError) as error:
                omissions.append(f'{path}: {error}')
        result.append({'image_id':visual['image_id'], 'dataset_index':visual['dataset_index'], 'assets':assets})
    return result, omissions


def validate_report(entry, root):
    """Accept actual complete records, not a progress header with planned counts."""
    summary_path = inside_file(entry["summary"], root, ".json")
    summary, digest = read_json(summary_path)
    if digest != checked_digest(entry["summary_sha256"]):
        raise ValueError("Final summary SHA differs from supervisor receipt")
    if not isinstance(summary, dict):
        raise ValueError("Summary must be an object")
    labels = KNOWN_REPORTS.get(summary.get("format"))
    if labels is None:
        raise ValueError("Unsupported summary contract; no success assumed")
    if (summary.get("status") != "completed" or summary.get("full_val_evaluated") is not True
            or summary.get("evaluated_images") != 3449
            or summary.get("completed_images_per_model") != {label: 3449 for label in labels}
            or summary.get("all_sources_unchanged") is not True
            or summary.get("observed_identity_verified") is not True
            or summary.get("training_performed") is not False
            or summary.get("dataset_role") != "validation"):
        raise ValueError("Not a verified completed 3449-frame validation")
    if (summary["format"] == "sam3-input-ve-boundary-validation-v1"
            and summary.get("parameters_unchanged_by_version_counter") is not True):
        raise ValueError("Input-VE report did not confirm unchanged model parameters")
    ids = summary.get("evaluated_image_ids", [])
    if len(ids) != 3449 or any(type(value) is not int for value in ids) or len(set(ids)) != 3449:
        raise ValueError("Missing/duplicate actual validation image identities")
    expected = {(identifier, side) for identifier in ids for side in SIDES}
    if set(summary.get("models", {})) != set(labels) or set(summary.get("record_files", {})) != set(labels):
        raise ValueError("Incomplete model or record inventory")
    record_receipts = {}
    for label in labels:
        record = summary["record_files"][label]
        path = inside_file(record["path"], root, ".json")
        if path.parent != summary_path.parent / "records":
            raise ValueError("Records must be in this summary's records directory")
        rows, row_digest = read_json(path)
        if row_digest != checked_digest(record["sha256"]) or not isinstance(rows, list) or len(rows) != 6898:
            raise ValueError("Record SHA/count mismatch")
        pairs = []
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("image_id")) is not int
                    or row.get("model") != label or row.get("identity_verified") is not True
                    or row.get("observed_coco_image_id") != row.get("image_id")):
                raise ValueError("Unchecked actual record identity/model")
            pairs.append((row["image_id"], row["prompt_key"]))
        if set(pairs) != expected or len(pairs) != len(set(pairs)):
            raise ValueError("Incomplete/duplicate actual image-side records")
        record_receipts[label] = {"path": str(path), "sha256": row_digest, "queries": len(rows)}
    report_path = inside_file(entry["report"], root, ".md")
    if not 0 < report_path.stat().st_size <= 8 * 1024**2:
        raise ValueError("Markdown report exceeds bounded copy size")
    report_digest = fingerprint(report_path)
    if report_digest != checked_digest(entry["report_sha256"]):
        raise ValueError("Report SHA differs from supervisor receipt")
    visuals, omissions = selected_visuals(summary, summary_path, root, labels)
    return {"summary": str(summary_path), "summary_sha256": digest,
            "report": str(report_path), "report_sha256": report_digest,
            "format": summary["format"], "actual_images_per_variant": 3449,
            "actual_records": record_receipts, "visuals": visuals, "visual_omissions": omissions}


def verified_run(snapshot):
    root, state = Path(snapshot["root"]), snapshot["state"]
    verified, rejected, seen = [], [], set()
    expected_reports = KNOWN_QUEUES.get(state.get("format"))
    reports, commands = state.get("verified_reports", {}), state.get("commands", {})
    if not isinstance(reports, dict) or not isinstance(commands, dict):
        rejected.append({"label": "queue", "error": "Malformed report/command inventory"})
    entries = list(reports.items()) if isinstance(reports, dict) else []
    for label, command in (commands.items() if isinstance(commands, dict) else []):
        # Input supervisor also stores a tiny smoke-probe receipt. It is not
        # one of the contracted full reports and must not be promoted to one.
        if expected_reports is not None and label not in expected_reports:
            continue
        if isinstance(command, dict) and command.get("status") == "completed" and isinstance(command.get("verified_result"), dict):
            entries.append((label, command["verified_result"]))
    for label, entry in entries:
        try:
            if not isinstance(entry, dict):
                raise ValueError("Invalid result receipt")
            candidate = validate_report(entry, root)
            if candidate["summary"] not in seen:
                seen.add(candidate["summary"])
                verified.append({"label": str(label), **candidate})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
            rejected.append({"label": str(label), "error": str(error)[:500]})
    # A partial queue may still contain one genuinely finished report. Expose
    # that report without upgrading the entire failed/running queue to success.
    actual_reports = {entry["label"]: entry["format"] for entry in verified}
    expected_comparison = QUEUE_TRAINING_COMPARISONS.get(state.get('format'))
    comparison = state.get('training_comparison', {})
    comparison_confirmed = (expected_comparison is not None and isinstance(comparison, dict)
                            and all(type(comparison.get(key)) is type(value) and comparison.get(key) == value
                                    for key, value in expected_comparison.items()))
    accepted_complete = (snapshot["status"] in SUCCESS and expected_reports is not None
                         and actual_reports == expected_reports and comparison_confirmed and not rejected)
    if snapshot["status"] in SUCCESS and not accepted_complete:
        rejected.append({"label": "queue", "error": (
            "Unknown queue contract; complete status is not independently certified" if expected_reports is None else
            f"Expected all reports {expected_reports}; actually verified {actual_reports}; "
            f"supervisor training-comparison receipt verified={comparison_confirmed}")})
    return {key: value for key, value in snapshot.items() if key != "state"} | {
        "validated_complete": accepted_complete, "verified_reports": verified, "rejected_reports": rejected}


def render_notes(runs, created, deadline, once):
    lines = ["# 夜间实验与晨间资料快照", "",
             f"生成时间：{created.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}。",
             f"发布截止：{deadline.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}；模式：{'单次快照' if once else '有界等待后快照'}。", "",
             "本文件仅记录此次明确指定任务，未启动模型/训练、未结束其他进程、未改标签或旧报告。分类包是生成时的副本，正文中的旧日期与历史运行状态仍保留；不会随服务器后续进度自动更新。", "",
             "## 实际状态与已验收报告", ""]
    for run in runs:
        lines += [f"### {Path(run['root']).name}", "", f"服务器目录：`{run['root']}`。",
                  f"状态：`{run['status']}`；完整报告验收：{'通过' if run['validated_complete'] else '未整体通过/未完成'}。", ""]
        for item in run["verified_reports"]:
            lines.append(f"- [{item['label']}：已核对三组各 3449 帧/6898 条左右记录]({item['report']})；报告 SHA256 `{item['report_sha256']}`。")
            for group in item['visuals']:
                lines.extend(['', f"预定渲染中的首/中/末样本：dataset index {group['dataset_index']}，image ID {group['image_id']}。未按 GT 或结果好坏筛图。", ''])
                lines.extend(f"- [{asset['title']}]({asset['path']})" for asset in group['assets'])
            if item['visual_omissions']:
                lines.extend(['', '本报告未能收录的图像：', ''])
                lines.extend(f'- {reason}' for reason in item['visual_omissions'])
        for item in run["rejected_reports"]:
            lines.append(f"- `{item['label']}` 未验收：{item['error']}。")
        if not run["verified_reports"]:
            lines.append("没有收录可严格验收的完整结果；不以计划帧数、部分进度或成功字样补写结果。")
        lines.append("")
    lines += ["## 解读与交付边界", "",
              "- 用户已确认 frame 0 的可见手是右手，而双侧参考为空：模型以 left_hand 检出该手，是已确认的错侧案例，缺标与错侧同时存在。其他帧尚未据此确认；不能把所有 reference-absent 输出都当真实误检，或把全部新增检出当正确。画面左/右不是解剖侧，冻结指标未重算。",
              "- frame 0 的追加人审依据见仓库 docs/nakehand-frame0-right-hand-review-2026-09-11.md；早期封存视频包内“手别待确认”是当时状态，不覆盖该历史副本。",
              "- 分割/边界结果是与当前辅助参考的一致性；保留缺标签、连续帧相关性与受限跨录像划分的说明，不宣称独立人工真值或跨人泛化。",
              "- 本 finalizer 核对状态、实际身份记录及摘要/报告 SHA，不重新运行 GPU 或全面重算所有指标；详细数值和协议见对应报告。",
              "- 新图只从各已验收报告既有渲染清单，按 dataset index 取首/中/末最多三组；保留 RGB、参考与各模型黑白预测分离文件，不叠色、不重新渲染。图像 SHA 在此次读取前后核对，不冒充历史 summary 已提供的图像指纹。ZIP 总 100 MiB、单文件 8 MiB、256 资产上限；不足时清单会明确省略，不伪装可离线打开。",
              "- 未完成/失败任务保留原恢复点；这里不重试、不续训、不改任何任务状态。",
              "- ZIP 与 SHA 仅已生成在服务器；没有实际复制到 Windows，也不承诺助手会在 9 点主动发消息。Windows 仍需由已有 SSH/scp 或人工复制取得。", ""]
    return "\n".join(lines)


def verify_receipt_sources(runs):
    """Keep the copied report tied to the records that were actually checked."""
    for run in runs:
        for report in run['verified_reports']:
            paths = [(report['summary'], report['summary_sha256']),
                     (report['report'], report['report_sha256'])]
            paths += [(entry['path'], entry['sha256']) for entry in report['actual_records'].values()]
            paths += [(asset['path'], asset['sha256']) for group in report['visuals'] for asset in group['assets']]
            for value, digest in paths:
                if fingerprint(value) != digest:
                    raise RuntimeError(f"Previously verified artifact changed: {value}")


def atomic_state(output, value):
    temporary = output / "finalizer-state.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "finalizer-state.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="One explicit run root; repeat up to four")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--docs-root", type=Path, help="physical docs root; defaults to project/docs or sibling docs")
    parser.add_argument("--deadline", required=True, help="Timezone-aware overall publication cutoff, at most nine hours ahead")
    parser.add_argument("--once", action="store_true", help="Publish one truthful snapshot without waiting")
    args = parser.parse_args(argv)
    try:
        args.deadline = datetime.fromisoformat(args.deadline)
        if args.deadline.tzinfo is None or not 120 < (args.deadline-datetime.now(timezone.utc)).total_seconds() <= 9*3600:
            raise ValueError("Require a timezone-aware deadline 120 seconds to nine hours ahead")
        args.project_root, args.output_dir = clean_path(args.project_root), clean_path(args.output_dir)
        args.run = [clean_path(path) for path in args.run]
        if not 1 <= len(args.run) <= 4 or len(set(args.run)) != len(args.run):
            raise ValueError("Require one to four distinct explicit run directories")
        if args.output_dir.exists() or args.output_dir.is_relative_to(args.project_root):
            raise ValueError("Output must be new and outside the repository")
        for root in args.run:
            if root == Path('/') or args.output_dir.is_relative_to(root) or root.is_relative_to(args.output_dir):
                raise ValueError("Output and watched runs must not contain each other")
        args.docs_root = resolve_docs_root(args.project_root, args.docs_root)
        if args.output_dir.is_relative_to(args.docs_root) or args.docs_root.is_relative_to(args.output_dir):
            raise ValueError("Output and documentation root must not contain each other")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def execute(args, *, now=None, sleep=None, run_command=None):
    now = now or (lambda: datetime.now(timezone.utc))
    sleep, run_command = sleep or time.sleep, run_command or subprocess.run
    package_script = clean_path(args.project_root/'scripts/package_review_docs.py')
    docs_root = resolve_docs_root(args.project_root, getattr(args, 'docs_root', None))
    if args.output_dir.is_relative_to(docs_root) or docs_root.is_relative_to(args.output_dir):
        raise ValueError("Output and documentation root must not contain each other")
    package_hash = fingerprint(package_script)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {"format": FORMAT, "status": "watching", "created_at": now().isoformat(),
             "deadline": args.deadline.isoformat(), "run_roots": [str(path) for path in args.run],
             "once": args.once, "gpu_work": False, "windows_transfer_performed": False,
             "poll_seconds": POLL_SECONDS, "package_source_sha256": package_hash,
             "source_docs_root": str(docs_root)}
    try:
        cutoff = args.deadline-timedelta(seconds=PUBLICATION_RESERVE_SECONDS)
        while True:
            snapshots = [snapshot_run(root) for root in args.run]
            state.update(observed=[{k:v for k,v in item.items() if k != 'state'} for item in snapshots], updated_at=now().isoformat())
            atomic_state(args.output_dir, state)
            if args.once or all(item['status'] in TERMINAL for item in snapshots) or now() >= cutoff:
                break
            sleep(min(POLL_SECONDS, max(0, (cutoff-now()).total_seconds())))
        verified = [verified_run(item) for item in snapshots]
        notes_root = args.output_dir/'notes'
        notes_root.mkdir()
        notes = notes_root/'OVERNIGHT_REVIEW.md'
        with notes.open('x', encoding='utf-8') as stream:
            stream.write(render_notes(verified, now(), args.deadline, args.once))
        reports = [notes] + [Path(item['report']) for value in verified for item in value['verified_reports']]
        verify_receipt_sources(verified)
        if fingerprint(package_script) != package_hash:
            raise RuntimeError("Packager changed since finalizer startup; no package executed")
        command = [sys.executable, str(package_script), '--repo-root', str(args.project_root),
                   '--docs-root', str(docs_root),
                   '--output-dir', str(args.output_dir/'bundle'), '--max-assets', '256']
        for root in [*args.run, notes_root]:
            command += ['--results-root', str(root)]
        for report_path in sorted(set(reports)):
            command += ['--report', str(report_path)]
        remaining = (args.deadline-now()).total_seconds()
        if remaining <= 1:
            raise TimeoutError("Publication cutoff reached; notes retained without ZIP")
        state.update(status='packaging', verified_runs=verified, notes=str(notes), command=command)
        atomic_state(args.output_dir, state)
        result = run_command(command, cwd=args.project_root, capture_output=True, text=True,
                             timeout=min(120, remaining), check=False)
        with (args.output_dir/'packaging.log').open('x', encoding='utf-8') as stream:
            stream.write(result.stdout+'\n'+result.stderr)
        if result.returncode or fingerprint(package_script) != package_hash:
            raise RuntimeError("Packaging failed or packager changed; see packaging.log")
        verify_receipt_sources(verified)
        receipt = json.loads(result.stdout)
        archive = inside_file(receipt['archive'], args.output_dir/'bundle', '.zip')
        if fingerprint(archive) != checked_digest(receipt['archive_sha256']):
            raise ValueError("Published archive differs from receipt SHA")
        state.update(status='snapshot_published', completed_at=now().isoformat(), receipt=receipt,
                     all_runs_validated_complete=all(item['validated_complete'] for item in verified))
        atomic_state(args.output_dir, state)
        print(json.dumps({key:state[key] for key in ('status','all_runs_validated_complete','notes','receipt')}, ensure_ascii=False))
        return 0
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', completed_at=now().isoformat())
        atomic_state(args.output_dir, state)
        raise


if __name__ == '__main__':
    raise SystemExit(execute(parse_args()))
