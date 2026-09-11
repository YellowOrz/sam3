"""Validate a completed bounded review ZIP, then atomically publish a data-only feed.

No network, source edits, executable payload, or credential handling. Existing
version directories are never changed; only the explicitly named latest feed is
replaced. Receipts and original archives remain available for audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from scripts.package_review_docs import FORMAT, clean_path, safe_member


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(bundle: Path, feed: Path) -> dict:
    bundle, feed = clean_path(bundle), clean_path(feed)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", bundle.name):
        raise ValueError("unsafe version name")
    safe_member(bundle.name)
    if feed.parent != bundle.parent or feed.name != "latest.json":
        raise ValueError("feed must be latest.json beside the immutable version directories")
    archive = clean_path(bundle / "review-docs.zip")
    receipt_path = clean_path(bundle / "receipt.json")
    if receipt_path.stat().st_size > 16384 or not 0 < archive.stat().st_size <= 200 * 1024**2:
        raise ValueError("receipt/archive size limit exceeded")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != FORMAT or receipt.get("source_files_verified_unchanged") is not True:
        raise ValueError("incomplete or unsupported package receipt")
    archive_bytes = archive.read_bytes()
    if (receipt.get("archive") != str(archive) or receipt.get("archive_bytes") != len(archive_bytes)
            or receipt.get("archive_sha256") != digest(archive_bytes)):
        raise ValueError("receipt does not match archive identity, size or SHA256")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
        names, total = set(), 0
        for entry in zipped.infolist():
            safe_member(entry.filename)
            total += entry.file_size
            if (entry.filename.casefold() in names or entry.is_dir() or entry.file_size > 16 * 1024**2
                    or ((entry.external_attr >> 16) & 0xF000) == 0xA000
                    or entry.external_attr & 0x400):
                raise ValueError("unsafe/duplicate/oversized archive entry")
            names.add(entry.filename.casefold())
        if total > 200 * 1024**2 or len(names) > 1024:
            raise ValueError("archive safety budget exceeded")
        manifest_bytes = zipped.read("manifest.json")
        if digest(manifest_bytes) != receipt.get("manifest_sha256"):
            raise ValueError("manifest SHA256 mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("format") != FORMAT or manifest.get("source_files_verified_unchanged") is not True:
            raise ValueError("incomplete manifest")
        expected = {"manifest.json"}
        for entry in manifest["files"]:
            member = safe_member(entry["path"])
            if member.casefold() in expected:
                raise ValueError("duplicate manifest entry")
            expected.add(member.casefold())
            payload = zipped.read(member)
            if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
                raise ValueError("manifest member integrity mismatch")
        if names != expected:
            raise ValueError("archive/manifest member set differs")
    result = dict(format="sam3-review-feed-v1", version=bundle.name, archive=str(archive),
                  sha256=digest(archive_bytes), bytes=len(archive_bytes),
                  manifest_sha256=digest(manifest_bytes),
                  created_at=datetime.now(timezone.utc).isoformat())
    # Validate one in-memory archive snapshot and reject concurrent source
    # changes before publishing its identity. Version directories are immutable.
    if clean_path(archive).read_bytes() != archive_bytes:
        raise RuntimeError("archive changed during verification; existing feed retained")
    if feed.exists():
        if feed.stat().st_size > 16384:
            raise ValueError("existing feed too large; refusing replacement")
        previous = json.loads(feed.read_text(encoding="utf-8"))
        if previous.get("format") != result["format"]:
            raise ValueError("refusing to replace unrelated existing file")
        if previous.get("version") == result["version"]:
            if any(previous.get(key) != result[key] for key in result if key != "created_at"):
                raise ValueError("existing version identity cannot change")
            return previous
    descriptor, temporary = tempfile.mkstemp(prefix=".latest-", suffix=".json", dir=feed.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    clean_path(feed)
    if clean_path(archive).read_bytes() != archive_bytes:
        raise RuntimeError("archive changed before feed publication; staging retained")
    os.replace(temporary, feed)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--feed", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.bundle, args.feed), indent=2))


if __name__ == "__main__":
    main()
