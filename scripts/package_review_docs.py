"""Create a bounded, portable review copy without changing original documents.

Only the selected docs root and individually named Markdown reports are included. No
dataset enumeration, checkpoint copy, network access, or source modification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, unquote, urlsplit
import zipfile


FORMAT = "sam3-portable-review-documents-v1"
DEFAULT_RESULTS_ROOTS = (
    "/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1",
    "/home/zhengyuxi/datasets/sam3-nakehand-experiments",
)
ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".dot", ".mp4"}
LINK = re.compile(r"(?P<image>!?)\[(?P<label>[^\]\n]*)\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>[^\s)]+))(?:\s+\"[^\"\n]*\")?\s*\)")
REFERENCE = re.compile(r"(?m)^(?P<prefix> {0,3}\[[^]\n]+\]:\s*)(?:<(?P<angle>[^>\n]+)>|(?P<plain>\S+))(?P<rest>[^\n]*)$")
CATEGORIES = {
    "00-start": "阅读入口与阶段进展",
    "01-goals": "目标、验收与学习 token 方案",
    "02-data": "数据审计、划分与处理",
    "03-results": "实测结果与对照实验",
    "04-mano-memory": "MANO、geometry 与 memory",
    "05-meeting": "导师／学长讨论与研究依据",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_member(name: str) -> str:
    """Portable ZIP names: reject traversal, ADS, drive and Windows aliases."""
    if not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive path: {name!r}")
    parts = name.split("/")
    for part in parts:
        if part in {"", ".", ".."} or part.endswith((".", " ")):
            raise ValueError(f"unsafe archive component: {part!r}")
        if re.search(r'[<>:"|?*\x00-\x1f]', part):
            raise ValueError(f"unsafe Windows component: {part!r}")
        if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part):
            raise ValueError(f"reserved Windows component: {part!r}")
    return name


def clean_path(path: Path) -> Path:
    """Refuse symlinks rather than following an apparent in-scope name."""
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"symlink paths are not allowed: {component}")
    return absolute


def within(path: Path, roots: list[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def resolve_docs_root(repo_root: Path, docs_root: Path | None = None) -> Path:
    """Select physical docs; allow only the exact local legacy-directory alias.

    No general symlink resolution is introduced. The optional repo/docs alias
    must point directly at the selected physical directory, not another alias.
    """
    repo = clean_path(repo_root)
    legacy = repo / "docs"
    selected = docs_root if docs_root is not None else (
        legacy if legacy.is_dir() and not legacy.is_symlink() else repo.parent / "docs")
    docs = clean_path(selected)
    if not docs.is_dir():
        raise ValueError(f"physical documentation directory is missing: {docs}")
    if legacy.is_symlink():
        direct_target = Path(os.path.abspath(legacy.parent / os.readlink(legacy)))
        if direct_target != docs:
            raise ValueError("legacy repo/docs alias must point directly at the selected docs root")
    return docs


def documentation_link_target(raw: str, source: Path, *, repo: Path,
                              docs: Path) -> tuple[Path | None, str]:
    """Remap only reading-copy links; never rewrite historical source bytes."""
    target, fragment = link_target(raw, source)
    legacy = repo / "docs"
    if target is None or docs == legacy:
        return target, fragment
    # An existing physical legacy tree is not an alias for another selected tree.
    if target.is_relative_to(legacy) and (legacy.is_symlink() or not legacy.exists()):
        return docs / target.relative_to(legacy), fragment
    # Historical ../scripts and ../sam3 links were authored inside repo/docs.
    # Preserve new links already addressing the actual repository. Other external
    # paths remain outside the asset whitelist and are explicitly server-only.
    if (source.is_relative_to(docs) and not Path(unquote(urlsplit(raw).path)).is_absolute()
            and not target.is_relative_to(docs) and not target.is_relative_to(repo)
            and target.is_relative_to(docs.parent)):
        relative = target.relative_to(docs.parent)
        if relative.parts and (repo / relative.parts[0]).exists():
            return repo / relative, fragment
    return target, fragment


def category(path: Path) -> str:
    name = path.name.lower()
    if "data-audits" in path.parts:
        return "02-data"
    if "meeting" in name or "research-notes" in name or name == "sam3-loss-reference.md":
        return "05-meeting"
    if any(word in name for word in ("mano", "memory", "geometry", "finetuning-stages")):
        return "04-mano-memory"
    if any(word in name for word in ("dataset", "data-preparation", "split", "external-test-protocol", "conversion-trust-audit", "dex-mask-resize")):
        return "02-data"
    if any(word in name for word in ("goal", "plan", "reliability")):
        return "01-goals"
    if any(word in name for word in ("results", "equivalence", "ablation", "numerical-audit")):
        return "03-results"
    return "00-start"


def compact_name(name: str, limit: int = 80) -> str:
    """Keep every member short enough for the default Windows staging root."""
    if len(name) <= limit:
        return name
    suffix = Path(name).suffix
    identity = sha256(name.encode())[:8]
    return name[:limit - len(suffix) - 9] + "-" + identity + suffix


def link_target(raw: str, source: Path) -> tuple[Path | None, str]:
    # Retain web links and document-local anchors. Linux path :line is provenance,
    # not a portable link fragment; code files remain explicit server-only links.
    if raw.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
        return None, ""
    split = urlsplit(raw)
    path_text = unquote(split.path)
    path_text = re.sub(r":\d+$", "", path_text)
    if not path_text:
        return None, ""
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return Path(os.path.abspath(candidate)), ("#" + split.fragment if split.fragment else "")


def non_code_segments(markdown: str):
    """Avoid rewriting commands and literal examples in fenced / inline code."""
    fenced = re.compile(r"(?ms)^ {0,3}(?:```[^\n]*\n.*?^ {0,3}```[^\n]*$|~~~[^\n]*\n.*?^ {0,3}~~~[^\n]*$)|`+[^`\n]*`+")
    start = 0
    for match in fenced.finditer(markdown):
        yield False, markdown[start:match.start()]
        yield True, match.group()
        start = match.end()
    yield False, markdown[start:]


def document_description(data: bytes) -> str:
    """Use the document's own title/prose, not an inferred experiment verdict."""
    markdown = data.decode("utf-8")
    outside_fences = re.sub(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*$", "", markdown)
    heading = re.search(r"(?m)^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", outside_fences)
    if heading:
        value = heading.group(1)
    else:
        value = next((line.strip() for line in outside_fences.splitlines()
                      if line.strip() and not re.match(r"^\s*(?:>|[-*+]\s|\d+[.)]\s|\||---|\[.*\]:)", line)), "未提供正文标题；请打开原文确认内容")
    value = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[`*_<>\[\]]", "", value)
    value = " ".join(value.split())
    return "原文主题：" + (value[:90] + "…" if len(value) > 90 else value)


def package(*, repo_root: Path, output_dir: Path, reports: list[Path],
            results_roots: list[Path], max_assets: int = 96,
            max_total_bytes: int = 100 * 1024**2, max_file_bytes: int = 8 * 1024**2,
            docs_root: Path | None = None) -> dict:
    repo = clean_path(repo_root)
    docs = resolve_docs_root(repo, docs_root)
    roots = [clean_path(root) for root in results_roots]
    if not docs.is_dir() or not roots:
        raise ValueError("physical docs and at least one approved results root required")
    if not 0 <= max_assets <= 256 or not 1 <= max_total_bytes <= 200 * 1024**2:
        raise ValueError("asset count / total size exceeds hard safety budget")
    if not 1 <= max_file_bytes <= 16 * 1024**2:
        raise ValueError("per-file budget exceeds 16 MiB")
    output = clean_path(output_dir)
    if output.exists():
        raise FileExistsError(f"new output directory required: {output}")
    if within(output, [docs, *roots]):
        raise ValueError("output must be outside documentation and approved source roots")
    # Discovery is confined to docs, never the dataset roots. A directory symlink
    # is an explicit error even if pathlib would otherwise omit its contents.
    document_paths = []
    for current, directories, files in os.walk(docs, followlinks=False):
        for name in directories + files:
            clean_path(Path(current) / name)
        for name in files:
            if name.lower().endswith(".md"):
                document_paths.append(Path(current) / name)
    changelog = clean_path(repo / "CHANGELOG.md")
    if changelog.is_file():
        document_paths.append(changelog)
    selected_reports = []
    for path in reports:
        path = clean_path(path)
        if path.suffix.lower() != ".md" or not within(path, roots) or not path.is_file():
            raise ValueError(f"report must be an explicit Markdown file in approved roots: {path}")
        selected_reports.append(path)
    sources: dict[Path, bytes] = {}
    destinations: dict[Path, str] = {}
    occupied: set[str] = set()
    total_bytes = 0
    omitted: list[dict] = []

    def add(path: Path, destination: str) -> None:
        nonlocal total_bytes
        path = clean_path(path)
        if path in sources:
            return
        if path != changelog and not within(path, [docs, *roots]):
            raise ValueError(f"source escaped approved roots: {path}")
        destination = safe_member(destination)
        if destination.casefold() in occupied:
            raise ValueError(f"case-insensitive destination collision: {destination}")
        if not path.is_file() or path.stat().st_size > max_file_bytes:
            raise ValueError(f"source missing, not regular, or exceeds per-file budget: {path}")
        data = path.read_bytes()
        if len(data) > max_file_bytes or total_bytes + len(data) > max_total_bytes:
            raise ValueError(f"source exceeds package byte budget: {path}")
        sources[path] = data
        destinations[path] = destination
        occupied.add(destination.casefold())
        total_bytes += len(data)

    for path in sorted(set(document_paths)):
        # Keep original names for easy recognition. Nested docs get a short
        # disambiguator without recreating long Linux paths on Windows.
        if path == changelog:
            add(path, "00-start/CHANGELOG.md")
            continue
        relative = path.relative_to(docs)
        name = path.name if len(relative.parts) == 1 else f"{sha256(str(relative).encode())[:8]}-{path.name}"
        add(path, f"{category(path)}/{compact_name(name)}")
    for path in sorted(set(selected_reports)):
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "-", path.parent.name)[:28]
        identity = sha256(str(path).encode())[:8]
        bucket = ("00-start" if path.name == "OVERNIGHT_REVIEW.md" else
                  "02-data" if ("preparation" in path.name.lower() or path.name.lower() == "readme.md") else "03-results")
        add(path, f"{bucket}/{compact_name(f'{prefix}-{identity}-{path.name}')}")

    # Explicit, small provenance/architecture assets are useful even when a
    # Markdown document only describes them rather than linking to them.
    forced_assets = [docs / "figures" / f"sam3-mano-architecture.{ext}" for ext in ("png", "svg", "dot")]
    forced_assets.append(docs / "figures/nakehand-manual-review-20260910/human-review.json")
    for name in ("human-review.json", "frozen-plan.json", "manifest.json"):
        forced_assets.append(docs / "data-audits/realsense-20260910/manual-review" / name)
    forced_assets.append(docs / "data-audits/realsense-20260910/diagnostic-review/human-review.json")
    forced_assets.append(docs / "data-audits/realsense-20260910/D4-temporal-review/human-review.json")
    forced_assets.append(docs / "data-audits/realsense-20260910/review-issues.json")
    forced_assets.append(docs / "data-audits/dexycb-conversion-20260910/pixel-sample-audit.json")
    forced_assets.append(docs / "data-audits/dexycb-conversion-20260910/supplementary-audit.json")
    forced_assets.append(docs / "dex-mask-resize-cpu-audit-20260910.json")
    asset_candidates = list(forced_assets)
    for path, data in list(sources.items()):
        for is_code, segment in non_code_segments(data.decode("utf-8")):
            if is_code:
                continue
            for pattern in (LINK, REFERENCE):
                for match in pattern.finditer(segment):
                    target, _ = documentation_link_target(
                        match.group("angle") or match.group("plain"), path, repo=repo, docs=docs)
                    if target and target.suffix.lower() in ASSET_EXTENSIONS:
                        asset_candidates.append(target)
    asset_count = 0
    for path in dict.fromkeys(asset_candidates):
        if path in sources:
            continue
        reason = None
        try:
            clean_path(path)
        except ValueError:
            reason = "symlink refused"
        if not reason and (not within(path, [docs, *roots]) or not path.is_file()):
            reason = "outside approved roots or unavailable"
        if not reason and asset_count >= max_assets:
            reason = "asset-count budget"
        if not reason and (path.stat().st_size > max_file_bytes or total_bytes + path.stat().st_size > max_total_bytes):
            reason = "byte budget"
        if reason:
            if path.exists() or path not in forced_assets:
                omitted.append({"source": str(path), "reason": reason})
            continue
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", path.name)[:70]
        add(path, f"figures/{sha256(str(path).encode())[:12]}-{name}")
        asset_count += 1

    server_only: list[dict] = []

    def rewritten(raw: str, source: Path) -> tuple[str | None, str | None]:
        target, fragment = documentation_link_target(raw, source, repo=repo, docs=docs)
        if target is None:
            return raw, None
        if target in destinations:
            relative = os.path.relpath(destinations[target], str(PurePosixPath(destinations[source]).parent)).replace(os.sep, "/")
            return quote(relative, safe="/._-~") + fragment, None
        server_only.append({"document": destinations[source], "original_target": raw,
                            "server_path": str(target), "reason": "not included in bounded document bundle"})
        return None, str(target)

    outputs: dict[str, bytes] = {}
    for source, data in sources.items():
        destination = destinations[source]
        if source.suffix.lower() == ".md":
            def replace_link(match):
                raw = match.group("angle") or match.group("plain")
                portable, server = rewritten(raw, source)
                if portable is not None:
                    return f'{match.group("image")}[{match.group("label")}]({portable})'
                label = match.group("label")
                return f"{label}（服务器路径，未打包：`{server}`）"

            def replace_reference(match):
                raw = match.group("angle") or match.group("plain")
                portable, server = rewritten(raw, source)
                if portable is not None:
                    return match.group("prefix") + portable + match.group("rest")
                # Preserve reference-label readability without manufacturing a
                # local clickable link to a nonexistent Windows file.
                reference_label = match.group("prefix").strip().removesuffix(":").strip("[]")
                return f"引用「{reference_label}」：服务器路径，未打包：`{server}`"

            text = "".join(segment if is_code else REFERENCE.sub(replace_reference, LINK.sub(replace_link, segment))
                           for is_code, segment in non_code_segments(data.decode("utf-8")))
            text = ("> 离线阅读副本：正文保留原记录时间；Linux 绝对路径及命令仍指服务器。"
                    "只有本包收录的文档／图片链接已转换为相对路径。以 00-start 的最新状态文档为准。\n\n" + text)
            data = text.encode("utf-8")
        outputs[destination] = data

    index = ["# SAM3 手物分割：分类离线资料包", "",
             "先读统一目标，再读最新进展和实测结果。历史文档保留当时状态，不代表后续实验仍未完成。",
             "本包是独立阅读副本；不覆盖 Windows 仓库代码，也不包含 checkpoint、原始视频或完整数据集。",
             "未打包路径明确保留为服务器路径，不能直接在 Windows 打开。", "",
             "源文档与输出文件的 SHA256 均记录在 manifest.json；ZIP 的外部 SHA256 用于传输核验。",
             "以下每份 Markdown 的一行简介摘自其原标题（无标题则取首条正文），不是重新推断的完成状态；当前状态先看夜间快照。", ""]
    for bucket, title in CATEGORIES.items():
        index.extend([f"## {bucket} — {title}", ""])
        for source, destination in sorted(destinations.items(), key=lambda item: item[1]):
            if destination.startswith(bucket + "/") and source.suffix.lower() == ".md":
                index.append(f"- [{source.name}]({quote(destination, safe='/._-~')}) — {document_description(sources[source])}")
        index.append("")
    index.extend(["## 图片与可移植性", "", f"收录 {asset_count} 个图片／图结构／小型人审回执；未收录项目见清单。",
                  "图像为原文件副本，没有重新着色或修改 mask。", ""])
    outputs["START_HERE.md"] = "\n".join(index).encode("utf-8")
    by_destination = {destination: source for source, destination in destinations.items()}
    entries = []
    for destination, data in sorted(outputs.items()):
        source = by_destination.get(destination)
        entries.append({"path": safe_member(destination), "bytes": len(data), "sha256": sha256(data),
                        "source": str(source) if source else None,
                        "source_sha256": sha256(sources[source]) if source else None})
    manifest = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_repo": str(repo), "source_docs_root": str(docs),
                "approved_results_roots": [str(path) for path in roots],
                "explicit_reports": [str(path) for path in selected_reports], "files": entries,
                "source_files_verified_unchanged": True, "server_only_links": server_only,
                "omitted_assets": omitted, "asset_count": asset_count,
                "limits": {"max_assets": max_assets, "max_total_bytes": max_total_bytes,
                           "max_file_bytes": max_file_bytes},
                "manifest_integrity": "manifest is covered by the external ZIP SHA256, not its own file list"}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    outputs["manifest.json"] = manifest_bytes
    if sum(map(len, outputs.values())) > max_total_bytes:
        raise ValueError("rewritten output plus manifest exceeds total package budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    # Keep a failed staging directory for inspection; never delete source data or
    # previously published versions, even when a concurrent source edit is found.
    content = staging / "review"
    content.mkdir()
    for destination, data in outputs.items():
        target = content / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    archive = staging / "review-docs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for destination in sorted(outputs):
            zipped.write(content / destination, safe_member(destination))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or set(zipped.namelist()) != set(outputs):
            raise ValueError("archive integrity or member-set mismatch")
        for member in zipped.infolist():
            safe_member(member.filename)
            if sha256(zipped.read(member)) != sha256(outputs[member.filename]):
                raise ValueError("archive content checksum mismatch")
    for source, data in sources.items():
        clean_path(source)
        if source.read_bytes() != data:
            raise RuntimeError(f"source changed while packaging; unpublished staging retained: {source}")
    if resolve_docs_root(repo, docs) != docs:
        raise RuntimeError("documentation root changed while packaging")
    zip_sha = sha256(archive.read_bytes())
    (staging / "review-docs.zip.sha256").write_text(f"{zip_sha}  review-docs.zip\n", encoding="ascii")
    receipt = {"format": FORMAT, "output_dir": str(output), "archive": str(output / archive.name),
               "archive_sha256": zip_sha, "archive_bytes": archive.stat().st_size,
               "manifest_sha256": sha256(manifest_bytes), "document_count": len(document_paths) + len(set(selected_reports)),
               "asset_count": asset_count, "source_files_verified_unchanged": True,
               "windows_transfer_status": "not attempted by this packager"}
    (staging / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        raise FileExistsError(f"output appeared during packaging; staging retained: {output}")
    staging.rename(output)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--docs-root", type=Path,
                        help="physical docs directory; default: repo/docs if physical, otherwise sibling docs")
    parser.add_argument("--output-dir", type=Path, required=True, help="new version directory, outside input roots")
    parser.add_argument("--report", type=Path, action="append", default=[], help="one explicitly selected .md report; repeatable")
    parser.add_argument("--results-root", type=Path, action="append", help="approved derived-results root; repeatable; never enumerated")
    parser.add_argument("--max-assets", type=int, default=96)
    parser.add_argument("--max-total-mib", type=int, default=100)
    parser.add_argument("--max-file-mib", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = package(repo_root=args.repo_root, output_dir=args.output_dir, reports=args.report,
                      docs_root=args.docs_root,
                      results_roots=args.results_root or [Path(path) for path in DEFAULT_RESULTS_ROOTS],
                      max_assets=args.max_assets, max_total_bytes=args.max_total_mib * 1024**2,
                      max_file_bytes=args.max_file_mib * 1024**2)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
