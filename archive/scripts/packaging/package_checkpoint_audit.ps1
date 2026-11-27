[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [string]$ProjectRoot = (Join-Path $PSScriptRoot "..\..\.."),

    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $outputPath = Join-Path $projectPath "outputs\packages\checkpoint_audits"
} else {
    $outputPath = [IO.Path]::GetFullPath($OutputRoot)
}

$selectionRoot = Join-Path $projectPath "Logs\EvaData\CheckpointSelection"
$selectionMatches = @(
    Get-ChildItem -LiteralPath $selectionRoot -Directory -Recurse |
        Where-Object { $_.Name -eq $RunId }
)
if ($selectionMatches.Count -ne 1) {
    throw "Expected exactly one checkpoint-selection directory for ${RunId}; found $($selectionMatches.Count)."
}
$selectionSource = $selectionMatches[0].FullName
$planSource = Join-Path $selectionSource "plan.json"
$selectedCsvSource = Join-Path $selectionSource "selected_checkpoints.csv"

foreach ($requiredPath in @($selectionSource, $planSource, $selectedCsvSource)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required checkpoint-selection artifact is missing: $requiredPath"
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleName = "checkpoint_audit_${RunId}_${timestamp}"
$stagePath = Join-Path $outputPath $bundleName
$archivePath = Join-Path $outputPath "$bundleName.zip"
$hashPath = "$archivePath.sha256.txt"

foreach ($newPath in @($stagePath, $archivePath, $hashPath)) {
    if (Test-Path -LiteralPath $newPath) {
        throw "Refusing to overwrite existing output: $newPath"
    }
}

New-Item -ItemType Directory -Force -Path $stagePath | Out-Null

function Get-SafeRelativePath {
    param([Parameter(Mandatory = $true)][string]$SourcePath)

    $resolvedSource = (Resolve-Path -LiteralPath $SourcePath).Path
    $baseUri = [Uri]($projectPath.TrimEnd("\") + "\")
    $sourceUri = [Uri]$resolvedSource
    $relativePath = [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($sourceUri).ToString()).Replace("/", "\")
    if ([IO.Path]::IsPathRooted($relativePath) -or $relativePath.StartsWith("..")) {
        throw "Source path is outside the project root: $resolvedSource"
    }
    return $relativePath
}

function Copy-ProjectFile {
    param([Parameter(Mandatory = $true)][string]$SourcePath)

    $relativePath = Get-SafeRelativePath -SourcePath $SourcePath
    $destinationPath = Join-Path $stagePath $relativePath
    $destinationParent = Split-Path -Parent $destinationPath
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $destinationPath -Force
}

function Copy-ProjectDirectory {
    param([Parameter(Mandatory = $true)][string]$SourcePath)

    $relativePath = Get-SafeRelativePath -SourcePath $SourcePath
    $destinationPath = Join-Path $stagePath $relativePath
    $destinationParent = Split-Path -Parent $destinationPath
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $destinationParent -Recurse -Force
}

Write-Output "[1/6] Copying checkpoint-selection plan, manifests, summaries, and logs..."
Copy-ProjectDirectory -SourcePath $selectionSource

$plan = Get-Content -LiteralPath $planSource -Raw | ConvertFrom-Json
$tasks = @($plan.tasks)
$incompleteTasks = @($tasks | Where-Object { $_.status -ne "complete" })
if ($incompleteTasks.Count -ne 0) {
    throw "Plan contains $($incompleteTasks.Count) task(s) that are not complete."
}

$resultDirectories = @(
    $tasks |
        ForEach-Object { $_.result_dir } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique
)
if ($resultDirectories.Count -ne $tasks.Count) {
    throw "Expected one unique result directory per task; tasks=$($tasks.Count), result_dirs=$($resultDirectories.Count)."
}

Write-Output "[2/6] Copying $($resultDirectories.Count) raw evaluation result directories..."
$copiedResultCount = 0
foreach ($resultDirectory in $resultDirectories) {
    if (-not (Test-Path -LiteralPath $resultDirectory -PathType Container)) {
        throw "Evaluation result directory is missing: $resultDirectory"
    }
    if ((Split-Path -Leaf $resultDirectory) -notlike "*ckptsel_${RunId}*") {
        throw "Evaluation directory does not match run id ${RunId}: $resultDirectory"
    }
    foreach ($requiredResultFile in @("summary_metrics.csv", "summary_metrics.json", "evaluation_manifest.json")) {
        $resultFile = Join-Path $resultDirectory $requiredResultFile
        if (-not (Test-Path -LiteralPath $resultFile -PathType Leaf)) {
            throw "Required evaluation artifact is missing: $resultFile"
        }
    }
    Copy-ProjectDirectory -SourcePath $resultDirectory
    $copiedResultCount += 1
    if (($copiedResultCount % 10) -eq 0 -or $copiedResultCount -eq $resultDirectories.Count) {
        Write-Output "  copied $copiedResultCount/$($resultDirectories.Count) result directories"
    }
}

Write-Output "[3/6] Copying selected models and training metadata..."
$selectedRows = @(Import-Csv -LiteralPath $selectedCsvSource)
if ($selectedRows.Count -eq 0) {
    throw "No selected checkpoints were found in $selectedCsvSource"
}

$selectedModelRecords = @()
foreach ($selectedRow in $selectedRows) {
    $modelSource = $selectedRow.model_absolute_path
    if (-not (Test-Path -LiteralPath $modelSource -PathType Leaf)) {
        throw "Selected model is missing: $modelSource"
    }

    $actualHash = (Get-FileHash -LiteralPath $modelSource -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = $selectedRow.model_sha256.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Selected model SHA256 mismatch: $modelSource"
    }

    $modelsDirectory = Split-Path -Parent $modelSource
    $trainingRunDirectory = Split-Path -Parent $modelsDirectory
    foreach ($metadataRelativePath in @("run.json", "config\config.ini")) {
        $metadataSource = Join-Path $trainingRunDirectory $metadataRelativePath
        if (-not (Test-Path -LiteralPath $metadataSource -PathType Leaf)) {
            throw "Training metadata is missing: $metadataSource"
        }
        Copy-ProjectFile -SourcePath $metadataSource
    }
    Copy-ProjectFile -SourcePath $modelSource

    $selectedModelRecords += [ordered]@{
        scene = $selectedRow.scene
        method = $selectedRow.method
        seed = [int]$selectedRow.seed
        checkpoint_type = $selectedRow.checkpoint_type
        checkpoint_step = $selectedRow.checkpoint_step
        project_relative_path = Get-SafeRelativePath -SourcePath $modelSource
        sha256 = $actualHash
    }
}

Write-Output "[4/6] Copying reproducibility code snapshot and historical summary tables..."
$codeRelativePaths = @(
    "archive\scripts\packaging\package_checkpoint_audit.ps1",
    "archive\scripts\selection\select_checkpoints.py",
    "scripts\evaluation\evaluate_single.py",
    "scripts\evaluation\evaluate_completed_runs.py",
    "scripts\training\train_parallel.py",
    "tools\airsim_instances.py",
    "tools\runtime_support.py",
    "pixi.toml",
    "pixi.lock"
)
$copiedCodePaths = @()
foreach ($codeRelativePath in $codeRelativePaths) {
    $codeSource = Join-Path $projectPath $codeRelativePath
    if (Test-Path -LiteralPath $codeSource -PathType Leaf) {
        Copy-ProjectFile -SourcePath $codeSource
        $copiedCodePaths += $codeRelativePath
    }
}

$historicalSummaryRelativePaths = @(
    "Logs\EvaData\FormalBatches\ENV2_TunnelObs_and_ENV3_TunnelActivate\summary_all_models.csv",
    "Logs\EvaData\FormalBatches\ENV1_Tunnel_and_ENV3_TunnelActivate\summary_all_models.csv"
)
$copiedHistoricalSummaries = @()
foreach ($summaryRelativePath in $historicalSummaryRelativePaths) {
    $summarySource = Join-Path $projectPath $summaryRelativePath
    if (Test-Path -LiteralPath $summarySource -PathType Leaf) {
        Copy-ProjectFile -SourcePath $summarySource
        $copiedHistoricalSummaries += $summaryRelativePath
    }
}

$bundleReadme = @"
UAV checkpoint-selection complete audit bundle

Run ID: $RunId
Created: $((Get-Date).ToString("o"))

Contents:
- Frozen checkpoint-selection plan and scenario manifests
- Screen, confirmation, selection, and independent test summaries
- All per-task raw evaluation result directories referenced by plan.json
- Selected model artifacts with run.json and config/config.ini
- Reproducibility code snapshot and dependency lock files
- Historical formal-evaluation summary tables when available
- BUNDLE_MANIFEST.csv containing SHA256 for every payload file

The absolute paths recorded in old manifests describe the source machine.
Use project-relative paths and SHA256 values when relocating this bundle.
"@
$bundleReadmePath = Join-Path $stagePath "BUNDLE_README.txt"
Set-Content -LiteralPath $bundleReadmePath -Value $bundleReadme -Encoding UTF8

$bundleInfo = [ordered]@{
    schema_version = 1
    run_id = $RunId
    created_at = (Get-Date).ToString("o")
    source_project_root = $projectPath
    task_count = $tasks.Count
    raw_result_directory_count = $resultDirectories.Count
    selected_model_count = $selectedRows.Count
    selected_models = $selectedModelRecords
    code_snapshot_files = $copiedCodePaths
    historical_summary_files = $copiedHistoricalSummaries
}
$bundleInfoPath = Join-Path $stagePath "BUNDLE_INFO.json"
$bundleInfo | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $bundleInfoPath -Encoding UTF8

Write-Output "[5/6] Hashing payload files and writing BUNDLE_MANIFEST.csv..."
$payloadFiles = @(Get-ChildItem -LiteralPath $stagePath -Recurse -File | Sort-Object FullName)
$manifestRows = foreach ($payloadFile in $payloadFiles) {
    $stageUri = [Uri]($stagePath.TrimEnd("\") + "\")
    $payloadUri = [Uri]$payloadFile.FullName
    [pscustomobject]@{
        relative_path = [Uri]::UnescapeDataString($stageUri.MakeRelativeUri($payloadUri).ToString()).Replace("/", "\")
        size_bytes = $payloadFile.Length
        sha256 = (Get-FileHash -LiteralPath $payloadFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifestPath = Join-Path $stagePath "BUNDLE_MANIFEST.csv"
$manifestRows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$stagedFiles = @(Get-ChildItem -LiteralPath $stagePath -Recurse -File)
$stagedBytes = ($stagedFiles | Measure-Object -Property Length -Sum).Sum

Write-Output "[6/6] Compressing $($stagedFiles.Count) files ($([math]::Round($stagedBytes / 1MB, 2)) MB)..."
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
Compress-Archive -Path (Join-Path $stagePath "*") -DestinationPath $archivePath -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipArchive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $zipEntryCount = @($zipArchive.Entries | Where-Object { -not [string]::IsNullOrEmpty($_.Name) }).Count
} finally {
    $zipArchive.Dispose()
}
if ($zipEntryCount -ne $stagedFiles.Count) {
    throw "ZIP entry count mismatch: staged=$($stagedFiles.Count), zip=$zipEntryCount"
}

$archiveFile = Get-Item -LiteralPath $archivePath
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath $hashPath -Value "$archiveHash  $($archiveFile.Name)" -Encoding ASCII -NoNewline

[pscustomobject]@{
    run_id = $RunId
    stage_path = $stagePath
    archive_path = $archivePath
    archive_size_bytes = $archiveFile.Length
    archive_size_mb = [math]::Round($archiveFile.Length / 1MB, 2)
    archive_sha256 = $archiveHash
    sha256_file = $hashPath
    staged_file_count = $stagedFiles.Count
    zip_entry_count = $zipEntryCount
    raw_result_directory_count = $resultDirectories.Count
    selected_model_count = $selectedRows.Count
} | ConvertTo-Json -Depth 4
