"""Static PowerShell guards and independent policy fixtures, not Windows execution.

The feed regex constants are read from the shipped script. The archive fixture
oracle documents its evidence contract on CPU; it does not emulate PowerShell
or establish Windows junction/native-argument behavior. A real parser is used
only if pwsh happens to be installed; Windows end-to-end acceptance remains due.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pull_review_docs_windows.ps1"
SOURCE = SCRIPT.read_text(encoding="utf-8")
FIELDS = {"format", "version", "archive", "sha256", "bytes", "manifest_sha256", "created_at"}
ROOT = "/home/reviewer/review-feed"


def constant(name):
    return re.search(rf"^\${name} = '([^']*)'$", SOURCE, re.MULTILINE)[1].replace(r"\z", r"\Z")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def remote_path(path):
    if not isinstance(path, str) or not re.fullmatch(constant("RemotePathPattern"), path):
        raise ValueError("remote path")
    if any(part in (".", "..") for part in path.split("/")):
        raise ValueError("traversal")


def relative_path(path):
    if not isinstance(path, str) or not path or re.search(r'[\\<>:"|?*\x00-\x1f]', path):
        raise ValueError("unsafe relative path")
    for part in path.split("/"):
        if (part in ("", ".", "..") or part.endswith((".", " "))
                or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", part, re.I)):
            raise ValueError("unsafe Windows component")


def feed_contract(raw, root=ROOT):
    """Independent test oracle; actual PowerShell enforcement is static-checked."""
    if not 0 < len(raw) <= 16 * 1024:
        raise ValueError("feed size")
    remote_path(root)
    feed = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(feed, dict) or set(feed) != FIELDS:
        raise ValueError("field set")
    if any(type(feed[key]) is not str for key in FIELDS - {"bytes"}):
        raise ValueError("string type")
    if feed["format"] != "sam3-review-feed-v1" or not re.fullmatch(constant("VersionPattern"), feed["version"]):
        raise ValueError("format/version")
    relative_path(feed["version"])
    remote_path(feed["archive"])
    if not feed["archive"].startswith(root + "/") or not feed["archive"].endswith(".zip"):
        raise ValueError("archive containment")
    if any(not re.fullmatch(constant("ShaPattern"), feed[key]) for key in ("sha256", "manifest_sha256")):
        raise ValueError("SHA")
    if type(feed["bytes"]) is not int or not 1 <= feed["bytes"] <= 200 * 1024 * 1024:
        raise ValueError("bytes")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", feed["created_at"]):
        raise ValueError("UTC timestamp")
    datetime.fromisoformat(feed["created_at"].replace("Z", "+00:00"))
    return feed


def publication_fixture():
    files = {"START_HERE.md": b"Reviewed data, not code.\n", "reports/result.md": "待真实结果。\n".encode()}
    manifest = {"format": "sam3-portable-review-documents-v1", "source_files_verified_unchanged": True,
                "files": [{"path": path, "bytes": len(data), "sha256": digest(data)} for path, data in files.items()]}
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    raw_zip = buffer.getvalue()
    feed = {"format": "sam3-review-feed-v1", "version": "v20260911-164500", "archive": ROOT + "/review.zip",
            "sha256": digest(raw_zip), "bytes": len(raw_zip), "manifest_sha256": digest(files["manifest.json"]),
            "created_at": "2026-09-11T08:45:00+00:00"}
    receipt = {"format": "sam3-portable-review-windows-transfer-v1", "source_alias": "lab_server",
               "source_archive": feed["archive"], "archive_sha256": feed["sha256"], "archive_bytes": feed["bytes"],
               "manifest_sha256": feed["manifest_sha256"], "destination": "C:\\review\\" + feed["version"],
               "verified_file_count": 2, "overwritten_existing_files": False}
    return feed, files, raw_zip, receipt


def publication_contract(feed, files, raw_zip, receipts, *, reparses=()):
    """Independent evidence oracle; no fake Windows runtime claim."""
    if reparses or len(receipts) != 1:
        raise ValueError("junction or missing/ambiguous receipt")
    receipt = receipts[0]
    expected = {"format": "sam3-portable-review-windows-transfer-v1", "source_alias": "lab_server",
                "source_archive": feed["archive"], "archive_sha256": feed["sha256"],
                "manifest_sha256": feed["manifest_sha256"], "destination": "C:\\review\\" + feed["version"],
                "overwritten_existing_files": False}
    if any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("receipt binding")
    if len(raw_zip) != feed["bytes"] or digest(raw_zip) != feed["sha256"]:
        raise ValueError("archive binding")
    if "archive_bytes" in receipt and (type(receipt["archive_bytes"]) is not int or receipt["archive_bytes"] != feed["bytes"]):
        raise ValueError("receipt bytes")
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        names = [name for name in archive.namelist() if name.casefold() == "manifest.json"]
        if names != ["manifest.json"] or digest(archive.read("manifest.json")) != feed["manifest_sha256"]:
            raise ValueError("ZIP manifest binding")
    if "manifest.json" not in files or digest(files["manifest.json"]) != feed["manifest_sha256"]:
        raise ValueError("local manifest binding")
    manifest = json.loads(files["manifest.json"])
    if type(receipt["verified_file_count"]) is not int or receipt["verified_file_count"] != len(manifest["files"]):
        raise ValueError("receipt count")
    expected_names = {"manifest.json"}
    for row in manifest["files"]:
        relative_path(row["path"])
        if row["path"].casefold() in expected_names:
            raise ValueError("duplicate manifest path")
        expected_names.add(row["path"].casefold())
        if row["path"] not in files or len(files[row["path"]]) != row["bytes"] or digest(files[row["path"]]) != row["sha256"]:
            raise ValueError("local contents")
    if {path.casefold() for path in files} != expected_names:
        raise ValueError("extra or missing file")
    return True


class PullStaticGuardsTest(unittest.TestCase):
    def test_four_parameters_are_mandatory_and_no_personal_defaults(self):
        parameter_block = SOURCE.split("Set-StrictMode", 1)[0]
        for name in ("ServerAlias", "RemoteFeed", "RemoteRoot", "DestinationRoot"):
            self.assertRegex(parameter_block, rf"(?s)\[Parameter\(Mandatory = \$true\)\]\s*(?:\[ValidatePattern\([^\n]+\)\]\s*)?\[string\] \${name}(?:,|\s*\))")
        self.assertNotIn("jixiegeming", SOURCE)
        self.assertNotIn("iipl_101", SOURCE)

    def test_strict_feed_shape_and_integer_limits_are_enforced(self):
        for fragment in ("$MaximumFeedBytes = 16KB", "$MaximumArchiveBytes = 200MB", "Match.Groups['key'].Captures",
                         "Keys.SetEquals", "Assert-Integer $Feed.bytes 1 $MaximumArchiveBytes", "$Value -isnot [int]",
                         "$Value -isnot [long]", "$Feed.archive.StartsWith($RemoteRoot + '/'", "System.StringComparison]::Ordinal"):
            self.assertIn(fragment, SOURCE)
        for field in FIELDS:
            self.assertIn("'" + field + "'", SOURCE)
        self.assertIn("$Match.Groups['value'].Captures[$Index].Value", SOURCE)
        self.assertIn("[long]::Parse($Token", SOURCE)
        self.assertIn("switch -CaseSensitive ($Escape.Groups[1].Value)", SOURCE)
        self.assertNotIn("$Feed = $Text | ConvertFrom-Json", SOURCE)

    def test_existing_version_never_skips_without_full_receipt_validation(self):
        self.assertEqual(SOURCE.count("Assert-PublishedVersion $Feed $PublishedPath $DestinationPath"), 2)
        self.assertRegex(SOURCE, r"(?s)if \(Test-Path -LiteralPath \$PublishedPath\) \{\s*Assert-PublishedVersion[^\n]+\n\s*return")
        for fragment in ("$MatchingReceipts.Count -ne 1", "'transfer-receipt.json'", "'review-docs.zip'",
                         "$Receipt.manifest_sha256 -cne $Feed.manifest_sha256", "$Receipt.overwritten_existing_files -isnot [bool]",
                         "$ActualFiles.SetEquals($ExpectedFiles)", "$ExpectedDirectories.Contains($Relative)",
                         "$ManifestEntries.Count -ne 1", "$ManifestSha -cne $Feed.manifest_sha256"):
            self.assertIn(fragment, SOURCE)
        self.assertNotRegex(SOURCE, r"(?i)\$Matches\s*=")

    def test_reparse_checks_precede_descent_and_native_transfer_is_noninteractive(self):
        self.assertIn("'^[A-Za-z]:[\\\\/]'", SOURCE)
        self.assertNotIn("Get-ChildItem -Recurse", SOURCE)
        block = SOURCE[SOURCE.index("foreach ($Child in Get-ChildItem"):]
        self.assertLess(block.index("ReparsePoint"), block.index("$Pending.Push"))
        scp = next(line for line in SOURCE.splitlines() if line.strip().startswith("& $ScpExecutable"))
        for option in ("-q", "BatchMode=yes", "StrictHostKeyChecking=yes", "ConnectTimeout=15", "ServerAliveInterval=15", "ServerAliveCountMax=2", " -- "):
            self.assertIn(option, scp)
        call = next(line for line in SOURCE.splitlines() if line.strip().startswith("& $SyncScript"))
        for option in ("-RemoteZip $Feed.archive", "-ExpectedSha256 $Feed.sha256", "-ExpectedArchiveBytes $Feed.bytes",
                       "-ExpectedManifestSha256 $Feed.manifest_sha256",
                       "-ServerAlias $ServerAlias", "-RemoteRoot $RemoteRoot", "-DestinationRoot $DestinationPath",
                       "-Version $Feed.version", "-BatchMode"):
            self.assertIn(option, call)
        self.assertIn("Join-Path $PSScriptRoot 'sync_review_docs_windows.ps1'", SOURCE)

    def test_no_remote_code_execution_deletion_scheduling_or_credential_changes(self):
        for token in ("Invoke-Expression", "Invoke-Command", "Start-Process", "Remove-Item", "rmdir", "del ",
                      "Set-ExecutionPolicy", "Register-ScheduledTask", "schtasks", "ssh-keygen", "ssh-add", "scp -O"):
            self.assertNotIn(token.casefold(), SOURCE.casefold())
        self.assertNotRegex(SOURCE, r"(?im)^\s*&?\s*ssh(?:\.exe)?\s")
        self.assertNotRegex(SOURCE, r"(?i)&\s*\$Feed\.")

    @unittest.skipUnless(shutil.which("pwsh"), "No pwsh: CPU static/contract checks only, not Windows runtime validation")
    def test_optional_powershell_parser_without_execution(self):
        source_literal = "'" + str(SCRIPT).replace("'", "''") + "'"
        code = "$t=$null;$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile(" + source_literal + ",[ref]$t,[ref]$e);if($e.Count){$e|Out-String|Write-Error;exit 1}"
        subprocess.run([shutil.which("pwsh"), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", code], check=True)


class PullPolicyFixtureTest(unittest.TestCase):
    def setUp(self):
        self.feed, self.files, self.raw_zip, self.receipt = publication_fixture()

    def test_valid_feed_and_complete_existing_evidence_contract(self):
        self.assertEqual(feed_contract(json.dumps(self.feed).encode()), self.feed)
        self.assertTrue(publication_contract(self.feed, self.files, self.raw_zip, [self.receipt]))

    def test_timestamp_and_json_escaped_strings_remain_strings(self):
        raw = json.dumps(self.feed).replace("/", r"\/").replace("sam3-review", r"sam3-\u0072eview")
        parsed = feed_contract(raw.encode())
        self.assertEqual(parsed, self.feed)
        self.assertIsInstance(parsed["created_at"], str)
        for value in ("2026-09-11", "v20260911"):
            self.assertEqual(feed_contract(json.dumps({**self.feed, "version": value}).encode())["version"], value)

    def test_missing_unknown_duplicate_and_case_colliding_feed_keys(self):
        for mode in ("missing", "unknown", "duplicate", "case"):
            data = deepcopy(self.feed)
            if mode == "missing":
                del data["sha256"]
            elif mode == "unknown":
                data["command"] = "do not execute"
            elif mode == "case":
                data["SHA256"] = data["sha256"]
            raw = json.dumps(data)
            if mode == "duplicate":
                raw = raw[:-1] + ', "bytes": 1}'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                feed_contract(raw.encode())

    def test_integer_size_sha_timestamp_and_format_limits(self):
        for key, value in (("bytes", True), ("bytes", "123"), ("bytes", 1.5), ("bytes", 0),
                           ("bytes", 200 * 1024 * 1024 + 1), ("sha256", "A" * 64),
                           ("manifest_sha256", "../bad"), ("created_at", "2026-09-11T08:00:00+08:00"),
                           ("created_at", "2026-02-31T00:00:00Z"), ("format", "other")):
            data = {**self.feed, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                feed_contract(json.dumps(data).encode())
        with self.assertRaises(ValueError):
            feed_contract(b" " * (16 * 1024 + 1))

    def test_archive_root_boundary_traversal_and_shell_characters(self):
        for path in (ROOT + "-sibling/a.zip", ROOT + "/../a.zip", ROOT + "//a.zip", ROOT + "/./a.zip",
                     ROOT + "/a.ZIP", ROOT + "/a.zip;echo", ROOT + "/a$(x).zip", "relative.zip",
                     ROOT + "/a\\b.zip", ROOT.upper() + "/a.zip", ROOT + "/a.zip\n"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                feed_contract(json.dumps({**self.feed, "archive": path}).encode())

    def test_windows_version_and_fully_qualified_destination_contract(self):
        for version in ("..", "v1.", "CON", "aux.txt", "x/y", "x\\y", "a" * 65):
            with self.subTest(version=version), self.assertRaises(ValueError):
                feed_contract(json.dumps({**self.feed, "version": version}).encode())
        for path in ("C:relative", "\\root-relative", "\\\\server\\share", "\\\\?\\C:\\review"):
            self.assertFalse(bool(re.match(r"^[A-Za-z]:[\\/]", path)) and PureWindowsPath(path).is_absolute())
        self.assertTrue(PureWindowsPath(r"C:\review").is_absolute())

    def test_directory_only_missing_ambiguous_or_mismatched_receipt_cannot_skip(self):
        for receipts in ([], [self.receipt, deepcopy(self.receipt)], [{**self.receipt, "archive_sha256": "0" * 64}],
                         [{**self.receipt, "destination": r"C:\review\other"}],
                         [{**self.receipt, "overwritten_existing_files": 0}], [{**self.receipt, "archive_bytes": 1}]):
            with self.subTest(receipts=receipts), self.assertRaises(ValueError):
                publication_contract(self.feed, self.files, self.raw_zip, receipts)

    def test_retained_zip_manifest_and_all_published_files_are_bound(self):
        for mode in ("zip_missing", "zip_changed", "manifest", "file_missing", "file_changed", "extra_hidden", "junction"):
            files, raw_zip = deepcopy(self.files), self.raw_zip
            if mode == "zip_missing":
                raw_zip = b""
            elif mode == "zip_changed":
                raw_zip = raw_zip[:-1] + bytes([raw_zip[-1] ^ 1])
            elif mode == "manifest":
                files["manifest.json"] += b" "
            elif mode == "file_missing":
                del files["START_HERE.md"]
            elif mode == "file_changed":
                files["START_HERE.md"] = b"changed"
            elif mode == "extra_hidden":
                files[".unlisted"] = b"extra"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                publication_contract(self.feed, files, raw_zip, [self.receipt], reparses=("reports",) if mode == "junction" else ())


if __name__ == "__main__":
    unittest.main()
