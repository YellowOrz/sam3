# One-shot pull of a data-only review feed. Schedule externally if desired.
# Requires an already configured SSH alias and trusted host key. No credential,
# execution-policy or scheduler changes are made; all existing versions survive.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('\A[A-Za-z0-9_][A-Za-z0-9_.-]*\z')]
    [string] $ServerAlias,
    [Parameter(Mandatory = $true)]
    [string] $RemoteFeed,
    [Parameter(Mandatory = $true)]
    [string] $RemoteRoot,
    [Parameter(Mandatory = $true)]
    [string] $DestinationRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RemotePathPattern = '\A/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\z'
$VersionPattern = '\A[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\z'
$ShaPattern = '\A[a-f0-9]{64}\z'
$MaximumFeedBytes = 16KB
$MaximumArchiveBytes = 200MB

function Assert-RemotePath([string] $Candidate) {
    if ($Candidate -notmatch $RemotePathPattern -or $Candidate.Split('/') -contains '.' -or
        $Candidate.Split('/') -contains '..') {
        throw 'Require a safe absolute POSIX path without traversal or shell metacharacters.'
    }
}

function Assert-RelativePath([string] $Candidate) {
    if ([string]::IsNullOrEmpty($Candidate) -or $Candidate.StartsWith('/') -or
        $Candidate.Contains('\') -or $Candidate -match '[<>:"|?*\x00-\x1f]') {
        throw 'Unsafe relative document path.'
    }
    foreach ($Part in $Candidate.Split('/')) {
        if ($Part -in @('', '.', '..') -or $Part.EndsWith('.') -or $Part.EndsWith(' ') -or
            $Part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$') {
            throw 'Unsafe Windows document path component.'
        }
    }
}

function Assert-NoReparseAncestor([string] $Candidate) {
    $Ancestor = [System.IO.Path]::GetFullPath($Candidate)
    while (-not [string]::IsNullOrEmpty($Ancestor)) {
        if (Test-Path -LiteralPath $Ancestor) {
            $Item = Get-Item -LiteralPath $Ancestor -Force
            if (($Item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing symlink/junction: $Ancestor"
            }
        }
        $Ancestor = [System.IO.Path]::GetDirectoryName($Ancestor)
    }
}

function Assert-Integer($Value, [long] $Minimum, [long] $Maximum) {
    if (($Value -isnot [int] -and $Value -isnot [long]) -or $Value -lt $Minimum -or $Value -gt $Maximum) {
        throw 'Expected an integer within the allowed byte/count range.'
    }
}

function Read-BoundedText([string] $Path, [long] $Limit) {
    Assert-NoReparseAncestor $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw 'Required local evidence file is missing.' }
    $Item = Get-Item -LiteralPath $Path -Force
    if ($Item.Length -le 0 -or $Item.Length -gt $Limit) { throw 'JSON evidence size limit exceeded.' }
    $Bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($Bytes.Length -gt $Limit) { throw 'JSON evidence grew while reading.' }
    $Text = [System.Text.UTF8Encoding]::new($false, $true).GetString($Bytes)
    if ($Text.Length -gt 0 -and $Text[0] -eq [char]0xFEFF) { $Text = $Text.Substring(1) }
    return $Text
}

function Read-Feed([string] $Path) {
    $Text = Read-BoundedText $Path $MaximumFeedBytes
    # A flat, fixed-schema JSON grammar rejects duplicate keys before the PS 5.1
    # JSON converter can silently replace them. Escaped property names are refused.
    $JsonString = '"(?:[^"\\\x00-\x1F]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"'
    $JsonValue = '(?:' + $JsonString + '|(?:0|[1-9][0-9]*))'
    $JsonPair = '"(?<key>[A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(?<value>' + $JsonValue + ')'
    $Match = [regex]::Match($Text, '\A\s*\{\s*' + $JsonPair + '(?:\s*,\s*' + $JsonPair + ')*\s*\}\s*\z',
        [System.Text.RegularExpressions.RegexOptions]::CultureInvariant, [TimeSpan]::FromSeconds(2))
    if (-not $Match.Success) { throw 'Feed must be flat JSON with string or unsigned-integer values.' }
    $Keys = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($Capture in $Match.Groups['key'].Captures) {
        if (-not $Keys.Add($Capture.Value)) { throw 'Duplicate feed key.' }
    }
    $Required = @('format', 'version', 'archive', 'sha256', 'bytes', 'manifest_sha256', 'created_at')
    if (-not $Keys.SetEquals([string[]] $Required)) { throw 'Unknown or missing feed field.' }
    # Decode only the already validated flat tokens. ConvertFrom-Json on newer
    # PowerShell auto-converts ISO strings to DateTime; -DateKind String is not
    # available in PS 5.1. Never reinterpret a feed value as executable text.
    $FeedValues = [ordered] @{}
    for ($Index = 0; $Index -lt $Match.Groups['key'].Captures.Count; $Index++) {
        $Name = $Match.Groups['key'].Captures[$Index].Value
        $Token = $Match.Groups['value'].Captures[$Index].Value
        if ($Token.StartsWith('"')) {
            $FeedValues[$Name] = [regex]::Replace($Token.Substring(1, $Token.Length - 2),
                '\\(["\\/bfnrt]|u[0-9a-fA-F]{4})', [System.Text.RegularExpressions.MatchEvaluator] {
                    param($Escape)
                    switch -CaseSensitive ($Escape.Groups[1].Value) {
                        '"' { return '"' }
                        '\' { return '\' }
                        '/' { return '/' }
                        'b' { return [string] [char] 8 }
                        'f' { return [string] [char] 12 }
                        'n' { return [string] [char] 10 }
                        'r' { return [string] [char] 13 }
                        't' { return [string] [char] 9 }
                        default { return [string] [char] [Convert]::ToInt32($Escape.Groups[1].Value.Substring(1), 16) }
                    }
                })
        }
        else {
            $FeedValues[$Name] = [long]::Parse($Token, [System.Globalization.NumberStyles]::None,
                [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    $Feed = [pscustomobject] $FeedValues
    foreach ($Name in @('format', 'version', 'archive', 'sha256', 'manifest_sha256', 'created_at')) {
        if ($Feed.$Name -isnot [string]) { throw 'Feed string field has the wrong type.' }
    }
    if ($Feed.format -cne 'sam3-review-feed-v1' -or $Feed.version -notmatch $VersionPattern -or
        $Feed.sha256 -cnotmatch $ShaPattern -or $Feed.manifest_sha256 -cnotmatch $ShaPattern) {
        throw 'Unknown feed format or invalid version/checksum.'
    }
    Assert-RelativePath $Feed.version
    Assert-RemotePath $Feed.archive
    if (-not $Feed.archive.StartsWith($RemoteRoot + '/', [System.StringComparison]::Ordinal) -or
        -not $Feed.archive.EndsWith('.zip', [System.StringComparison]::Ordinal)) {
        throw 'Feed archive must be a .zip strictly inside RemoteRoot.'
    }
    Assert-Integer $Feed.bytes 1 $MaximumArchiveBytes
    if ($Feed.created_at -notmatch '\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)\z') {
        throw 'Feed created_at must be an ISO8601 UTC string.'
    }
    [void] [DateTimeOffset]::Parse($Feed.created_at, [System.Globalization.CultureInfo]::InvariantCulture)
    return $Feed
}

function Assert-PublishedVersion($Feed, [string] $PublishedPath, [string] $DestinationPath) {
    Assert-NoReparseAncestor $PublishedPath
    if (-not (Test-Path -LiteralPath $PublishedPath -PathType Container)) { throw 'Published version is not a directory.' }
    $MatchingReceipts = @()
    # Receipts belong only to the existing sync script's direct staging folders.
    # Never recursively search arbitrary directories for a plausible receipt.
    foreach ($Directory in Get-ChildItem -LiteralPath $DestinationPath -Directory -Force) {
        if ($Directory.Name -notmatch '^\.transfer-[a-fA-F0-9]{32}$') { continue }
        Assert-NoReparseAncestor $Directory.FullName
        $ReceiptPath = Join-Path $Directory.FullName 'transfer-receipt.json'
        if (-not (Test-Path -LiteralPath $ReceiptPath)) { continue }
        Assert-NoReparseAncestor $ReceiptPath
        try {
            $Receipt = Read-BoundedText $ReceiptPath 64KB | ConvertFrom-Json
            if ($null -eq $Receipt -or $null -eq $Receipt.PSObject.Properties['destination'] -or
                $Receipt.destination -isnot [string] -or $Receipt.destination -notmatch '^[A-Za-z]:[\\/]') { continue }
            $ReceiptDestination = [System.IO.Path]::GetFullPath($Receipt.destination)
        }
        catch { continue } # Preserve unrelated interrupted receipts; never accept them as evidence.
        if (-not $ReceiptDestination.Equals($PublishedPath, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
        $MatchingReceipts += [pscustomobject] @{ Receipt = $Receipt; TransferPath = $Directory.FullName }
    }
    if ($MatchingReceipts.Count -ne 1) { throw 'Existing version requires exactly one matching transfer receipt; nothing overwritten.' }
    $Receipt = $MatchingReceipts[0].Receipt
    if ($Receipt.format -cne 'sam3-portable-review-windows-transfer-v1' -or
        $Receipt.overwritten_existing_files -isnot [bool] -or $Receipt.overwritten_existing_files -ne $false -or
        $Receipt.source_alias -cne $ServerAlias -or $Receipt.source_archive -cne $Feed.archive -or
        $Receipt.archive_sha256 -cne $Feed.sha256 -or $Receipt.manifest_sha256 -cne $Feed.manifest_sha256) {
        throw 'Transfer receipt does not match this feed.'
    }
    if ($null -ne $Receipt.PSObject.Properties['archive_bytes']) {
        Assert-Integer $Receipt.archive_bytes 1 $MaximumArchiveBytes
        if ($Receipt.archive_bytes -ne $Feed.bytes) { throw 'Receipt archive length differs from feed.' }
    }
    $ArchivePath = Join-Path $MatchingReceipts[0].TransferPath 'review-docs.zip'
    Assert-NoReparseAncestor $ArchivePath
    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf) -or
        (Get-Item -LiteralPath $ArchivePath -Force).Length -ne $Feed.bytes -or
        (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Feed.sha256) {
        throw 'Retained archive bytes/checksum do not match feed.'
    }
    $ManifestPath = Join-Path $PublishedPath 'manifest.json'
    $ManifestText = Read-BoundedText $ManifestPath 16MB
    if ((Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Feed.manifest_sha256) {
        throw 'Published manifest differs from feed.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $ManifestEntries = @($Archive.Entries | Where-Object { $_.FullName -ieq 'manifest.json' })
        if ($ManifestEntries.Count -ne 1 -or $ManifestEntries[0].FullName -cne 'manifest.json' -or
            $ManifestEntries[0].Length -le 0 -or $ManifestEntries[0].Length -gt 16MB) {
            throw 'Archive requires exactly one bounded manifest.json.'
        }
        $Reader = $ManifestEntries[0].Open()
        $Hasher = [System.Security.Cryptography.SHA256]::Create()
        try { $ManifestSha = [BitConverter]::ToString($Hasher.ComputeHash($Reader)).Replace('-', '').ToLowerInvariant() }
        finally { $Reader.Dispose(); $Hasher.Dispose() }
        if ($ManifestSha -cne $Feed.manifest_sha256) { throw 'Archive manifest differs from feed and published manifest.' }
    }
    finally { $Archive.Dispose() }
    $Manifest = $ManifestText | ConvertFrom-Json
    if ($Manifest.format -cne 'sam3-portable-review-documents-v1' -or
        $Manifest.source_files_verified_unchanged -isnot [bool] -or $Manifest.source_files_verified_unchanged -ne $true -or
        $Manifest.files -isnot [array] -or $Manifest.files.Count -gt 1023) { throw 'Incomplete or malformed document manifest.' }
    Assert-Integer $Receipt.verified_file_count 0 1023
    if ($Receipt.verified_file_count -ne $Manifest.files.Count) { throw 'Receipt file count mismatch.' }
    $ExpectedFiles = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    [void] $ExpectedFiles.Add('manifest.json')
    $ExpectedDirectories = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    [long] $TotalBytes = (Get-Item -LiteralPath $ManifestPath -Force).Length
    foreach ($Entry in $Manifest.files) {
        if ($Entry.path -isnot [string] -or $Entry.sha256 -isnot [string] -or $Entry.sha256 -cnotmatch $ShaPattern) {
            throw 'Invalid manifest path/checksum.'
        }
        Assert-RelativePath $Entry.path
        if (-not $ExpectedFiles.Add($Entry.path)) { throw 'Duplicate/case-colliding manifest file.' }
        Assert-Integer $Entry.bytes 0 16MB
        $TotalBytes += $Entry.bytes
        if ($TotalBytes -gt 200MB) { throw 'Manifest exceeds uncompressed safety limit.' }
        $FilePath = Join-Path $PublishedPath $Entry.path.Replace('/', '\')
        Assert-NoReparseAncestor $FilePath
        if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf) -or
            (Get-Item -LiteralPath $FilePath -Force).Length -ne $Entry.bytes -or
            (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Entry.sha256) {
            throw "Published file missing or changed: $($Entry.path)"
        }
        $Parts = $Entry.path.Split('/')
        for ($Index = 1; $Index -lt $Parts.Length; $Index++) {
            [void] $ExpectedDirectories.Add(($Parts[0..($Index - 1)] -join '/'))
        }
    }
    $ActualFiles = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $Pending = [System.Collections.Generic.Stack[string]]::new()
    $Pending.Push($PublishedPath)
    while ($Pending.Count -gt 0) {
        $Current = $Pending.Pop()
        Assert-NoReparseAncestor $Current
        foreach ($Child in Get-ChildItem -LiteralPath $Current -Force) {
            if (($Child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Published tree contains a symlink/junction.' }
            $Relative = $Child.FullName.Substring($PublishedPath.Length + 1).Replace('\', '/')
            if ($Child.PSIsContainer) {
                if (-not $ExpectedDirectories.Contains($Relative)) { throw 'Published tree has an unlisted directory.' }
                $Pending.Push($Child.FullName)
            }
            elseif (-not $ActualFiles.Add($Relative) -or -not $ExpectedFiles.Contains($Relative)) {
                throw 'Published tree has an unlisted or duplicate file.'
            }
        }
    }
    if (-not $ActualFiles.SetEquals($ExpectedFiles)) { throw 'Published file coverage differs from manifest.' }
}

if ($env:OS -ne 'Windows_NT') { throw 'Run this script on Windows with PowerShell and OpenSSH already configured.' }
Assert-RemotePath $RemoteRoot
Assert-RemotePath $RemoteFeed
if ($DestinationRoot -notmatch '^[A-Za-z]:[\\/]') { throw 'DestinationRoot must be a fully qualified local drive path, not UNC or drive-relative.' }
$DestinationPath = [System.IO.Path]::GetFullPath($DestinationRoot).TrimEnd([char[]] '\/')
if ($DestinationPath -match '^[A-Za-z]:$') { throw 'Refusing a drive root as DestinationRoot.' }
Assert-NoReparseAncestor $DestinationPath
$SyncScript = Join-Path $PSScriptRoot 'sync_review_docs_windows.ps1'
Assert-NoReparseAncestor $SyncScript
if (-not (Test-Path -LiteralPath $SyncScript -PathType Leaf)) { throw 'Local ZIP verification script is missing.' }
$ScpExecutable = (Get-Command scp -CommandType Application -ErrorAction Stop).Source
[System.IO.Directory]::CreateDirectory($DestinationPath) | Out-Null
$FeedTransfer = Join-Path $DestinationPath ('.pull-feed-' + [guid]::NewGuid().ToString('N'))
[System.IO.Directory]::CreateDirectory($FeedTransfer) | Out-Null
$FeedPath = Join-Path $FeedTransfer 'feed.json'

try {
    # Size is checked after scp completes, before parsing. This is an acceptance
    # limit, not a promise that network transfer can never exceed 16 KiB.
    & $ScpExecutable -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -- ($ServerAlias + ':' + $RemoteFeed) $FeedPath
    if ($LASTEXITCODE -ne 0) { throw "Feed scp failed with exit code $LASTEXITCODE" }
    $Feed = Read-Feed $FeedPath
    $PublishedPath = Join-Path $DestinationPath $Feed.version
    if (Test-Path -LiteralPath $PublishedPath) {
        Assert-PublishedVersion $Feed $PublishedPath $DestinationPath
        return # Quiet only after receipt, retained ZIP, manifest and every file pass.
    }
    Assert-NoReparseAncestor $DestinationPath
    & $SyncScript -RemoteZip $Feed.archive -ExpectedSha256 $Feed.sha256 -ExpectedArchiveBytes $Feed.bytes -ExpectedManifestSha256 $Feed.manifest_sha256 -ServerAlias $ServerAlias -RemoteRoot $RemoteRoot -DestinationRoot $DestinationPath -Version $Feed.version -BatchMode
    Assert-PublishedVersion $Feed $PublishedPath $DestinationPath
    Write-Output "Feed version verified: $($Feed.version)"
}
catch {
    Write-Warning "Pull did not verify a complete version; existing data and feed evidence are retained: $FeedTransfer"
    throw
}
