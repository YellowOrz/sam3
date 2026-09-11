# Download a reviewed data archive only. Never executes server-side commands or
# code from the archive. Run this locally with the existing OpenSSH configuration.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $RemoteZip,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-fA-F0-9]{64}$')]
    [string] $ExpectedSha256,
    [ValidatePattern('^[A-Za-z0-9_][A-Za-z0-9_.-]*$')]
    [string] $ServerAlias = 'iipl_101',
    [string] $DestinationRoot = 'C:\Users\jixiegeming\Desktop\paper_review\docs\overnight-review-2026-09-11',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string] $Version = (Get-Date -Format 'yyyyMMdd-HHmmss')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-ArchivePath([string] $ArchiveName) {
    if ([string]::IsNullOrEmpty($ArchiveName) -or $ArchiveName.StartsWith('/') -or
        $ArchiveName.Contains('\') -or $ArchiveName -match '[<>:"|?*\x00-\x1f]') {
        throw "Unsafe ZIP path: $ArchiveName"
    }
    foreach ($ArchivePart in $ArchiveName.Split('/')) {
        if ($ArchivePart -in @('', '.', '..') -or $ArchivePart.EndsWith('.') -or
            $ArchivePart.EndsWith(' ') -or $ArchivePart -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$') {
            throw "Unsafe ZIP path component: $ArchivePart"
        }
    }
}

function Assert-NoReparseAncestor([string] $CandidatePath) {
    $AncestorPath = [System.IO.Path]::GetFullPath($CandidatePath)
    while (-not [string]::IsNullOrEmpty($AncestorPath)) {
        if (Test-Path -LiteralPath $AncestorPath) {
            $AncestorItem = Get-Item -LiteralPath $AncestorPath -Force
            if (($AncestorItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing symlink/junction destination ancestor: $AncestorPath"
            }
        }
        $AncestorPath = [System.IO.Path]::GetDirectoryName($AncestorPath)
    }
}

if ($env:OS -ne 'Windows_NT') { throw 'Run this script in Windows PowerShell / PowerShell on Windows.' }
if ($RemoteZip -notmatch '^/home/zhengyuxi/(datasets|projects)/[A-Za-z0-9_./-]+\.zip$' -or
    $RemoteZip.Split('/') -contains '..' -or $RemoteZip.Contains('//')) {
    throw 'RemoteZip must be a safe absolute .zip path under the approved server datasets/projects directories.'
}
Assert-ArchivePath $Version
$ScpExecutable = (Get-Command scp -CommandType Application -ErrorAction Stop).Source
$DestinationPath = [System.IO.Path]::GetFullPath($DestinationRoot)
if (-not [System.IO.Path]::IsPathRooted($DestinationRoot) -or $DestinationPath.StartsWith('\\')) {
    throw 'DestinationRoot must be an absolute local Windows directory, not a network share.'
}
Assert-NoReparseAncestor $DestinationPath
$PublishedPath = Join-Path $DestinationPath $Version
if (Test-Path -LiteralPath $PublishedPath) { throw "Version already exists; nothing overwritten: $PublishedPath" }
[System.IO.Directory]::CreateDirectory($DestinationPath) | Out-Null
$TransferPath = Join-Path $DestinationPath ('.transfer-' + [guid]::NewGuid().ToString('N'))
[System.IO.Directory]::CreateDirectory($TransferPath) | Out-Null
$DownloadPath = Join-Path $TransferPath 'review-docs.zip'
$ExtractionPath = Join-Path $TransferPath 'verified-content'
[System.IO.Directory]::CreateDirectory($ExtractionPath) | Out-Null

try {
    # No remote shell command, installation, SSH configuration change or credential
    # collection. The existing local SSH alias handles authentication normally.
    & $ScpExecutable -- ($ServerAlias + ':' + $RemoteZip) $DownloadPath
    if ($LASTEXITCODE -ne 0) { throw "scp failed with exit code $LASTEXITCODE" }
    $DownloadedSha = (Get-FileHash -LiteralPath $DownloadPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($DownloadedSha -ne $ExpectedSha256.ToLowerInvariant()) { throw 'Downloaded ZIP SHA256 does not match the independently supplied expected value.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $ArchiveHandle = [System.IO.Compression.ZipFile]::OpenRead($DownloadPath)
    try {
        $ArchiveNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
        [long] $UncompressedBytes = 0
        if ($ArchiveHandle.Entries.Count -gt 1024) { throw 'Archive exceeds the 1024-file safety limit.' }
        foreach ($ArchiveEntry in $ArchiveHandle.Entries) {
            Assert-ArchivePath $ArchiveEntry.FullName
            if (-not $ArchiveNames.Add($ArchiveEntry.FullName)) { throw 'Duplicate/case-colliding ZIP member.' }
            # Unix symlink mode and DOS reparse-point flags are never accepted.
            $UnixType = (($ArchiveEntry.ExternalAttributes -shr 16) -band 0xF000)
            if ($UnixType -eq 0xA000 -or ($ArchiveEntry.ExternalAttributes -band 0x400) -ne 0) {
                throw 'Archive symlink/reparse entry refused.'
            }
            if ($ArchiveEntry.Length -gt 16MB -or $ArchiveEntry.Length -lt 0) { throw 'Archive per-file size limit exceeded.' }
            $UncompressedBytes += $ArchiveEntry.Length
            if ($UncompressedBytes -gt 200MB) { throw 'Archive exceeds 200 MiB uncompressed safety limit.' }
            $ExtractedFilePath = [System.IO.Path]::GetFullPath((Join-Path $ExtractionPath $ArchiveEntry.FullName.Replace('/', '\')))
            $ExpectedPrefix = $ExtractionPath.TrimEnd('\') + '\'
            if (-not $ExtractedFilePath.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Archive member escaped the staging directory.'
            }
            # Conservative Windows PowerShell 5.1 path compatibility. Abort before
            # writing any entry rather than depending on registry long-path policy.
            $FinalFilePath = Join-Path $PublishedPath $ArchiveEntry.FullName.Replace('/', '\')
            if ($ExtractedFilePath.Length -ge 260 -or $FinalFilePath.Length -ge 260) {
                throw 'Path exceeds the portable Windows limit; choose a shorter DestinationRoot or repack shorter names.'
            }
        }
        if (-not $ArchiveNames.Contains('manifest.json')) { throw 'Missing manifest.json.' }
        foreach ($ArchiveEntry in $ArchiveHandle.Entries) {
            $ExtractedFilePath = Join-Path $ExtractionPath $ArchiveEntry.FullName.Replace('/', '\')
            [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($ExtractedFilePath)) | Out-Null
            $EntryReader = $ArchiveEntry.Open()
            $EntryWriter = [System.IO.File]::Open($ExtractedFilePath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
            try { $EntryReader.CopyTo($EntryWriter) }
            finally { $EntryWriter.Dispose(); $EntryReader.Dispose() }
        }
    }
    finally { $ArchiveHandle.Dispose() }
    $ManifestPath = Join-Path $ExtractionPath 'manifest.json'
    $ManifestData = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($ManifestData.format -ne 'sam3-portable-review-documents-v1' -or $ManifestData.source_files_verified_unchanged -ne $true) {
        throw 'Unknown or incomplete document manifest.'
    }
    $ManifestNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    [void] $ManifestNames.Add('manifest.json')
    foreach ($ManifestEntry in $ManifestData.files) {
        Assert-ArchivePath $ManifestEntry.path
        if (-not $ManifestNames.Add($ManifestEntry.path)) { throw 'Duplicate manifest path.' }
        if ($ManifestEntry.sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid manifest checksum.' }
        $VerifiedFilePath = Join-Path $ExtractionPath $ManifestEntry.path.Replace('/', '\')
        if (-not (Test-Path -LiteralPath $VerifiedFilePath -PathType Leaf)) { throw 'Manifest file missing from archive.' }
        if ((Get-Item -LiteralPath $VerifiedFilePath).Length -ne $ManifestEntry.bytes -or
            (Get-FileHash -LiteralPath $VerifiedFilePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ManifestEntry.sha256) {
            throw "Manifest size/checksum mismatch: $($ManifestEntry.path)"
        }
    }
    if (-not $ManifestNames.SetEquals($ArchiveNames)) { throw 'ZIP contains unlisted files or manifest has missing files.' }
    Assert-NoReparseAncestor $DestinationPath
    if (Test-Path -LiteralPath $PublishedPath) { throw 'Version appeared during transfer; refusing to overwrite.' }
    # Same-volume rename publishes only a fully verified directory, and fails if
    # the destination already exists; does not merge with an existing folder.
    [System.IO.Directory]::Move($ExtractionPath, $PublishedPath)
    $TransferReceipt = [ordered]@{
        format = 'sam3-portable-review-windows-transfer-v1'
        completed_at = (Get-Date).ToUniversalTime().ToString('o')
        source_alias = $ServerAlias
        source_archive = $RemoteZip
        archive_sha256 = $DownloadedSha
        manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $PublishedPath 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        destination = $PublishedPath
        verified_file_count = $ManifestData.files.Count
        overwritten_existing_files = $false
    }
    $TransferReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $TransferPath 'transfer-receipt.json') -Encoding UTF8
    Write-Output "Verified documents published: $PublishedPath"
    Write-Output "Start reading: $(Join-Path $PublishedPath 'START_HERE.md')"
    Write-Output "ZIP and transfer receipt retained: $TransferPath"
}
catch {
    Write-Warning "Transfer not fully completed. Existing versions were not overwritten; staging retained for inspection: $TransferPath"
    throw
}
