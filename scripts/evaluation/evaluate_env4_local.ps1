[CmdletBinding()]
param(
    [switch]$Execute,
    [ValidateSet("all", "screen", "test")]
    [string]$Stage = "all",
    [string]$Resume = "",
    [string]$TrainingRoot = "Logs\TrainData\Runs",
    [string]$OutputRoot = "Logs\EvaData\CheckpointSelection\ENV4_TunnelMixed",
    [string]$UeExe = "D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe",
    [string]$SettingsTemplate = "C:\Users\31732\Documents\AirSim\settings.json",
    [ValidateRange(0, 3600)]
    [int]$StartupWait = 60,
    [switch]$NoLaunchAirSim
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ports = @(41451, 41461, 41471)
$maxParallel = 3
$pixiCommand = Get-Command pixi -ErrorAction Stop

if ($Stage -eq "test" -and [string]::IsNullOrWhiteSpace($Resume)) {
    throw "Stage=test requires -Resume with a frozen plan ID, directory, plan.json, or 'latest'."
}

$resolvedTrainingRoot = Join-Path $repoRoot $TrainingRoot
if (-not (Test-Path -LiteralPath $resolvedTrainingRoot -PathType Container)) {
    throw "Training root not found: $resolvedTrainingRoot"
}

if ($Execute -and -not $NoLaunchAirSim) {
    if (-not (Test-Path -LiteralPath $UeExe -PathType Leaf)) {
        throw "Tunnel UE executable not found: $UeExe"
    }
    if (-not (Test-Path -LiteralPath $SettingsTemplate -PathType Leaf)) {
        throw "AirSim settings template not found: $SettingsTemplate"
    }
}

$arguments = @(
    "run",
    "python",
    "-m",
    "scripts.selection.select_and_evaluate_env4",
    "--training-root", $TrainingRoot,
    "--output-root", $OutputRoot,
    "--candidate-mode", "training_peaks",
    "--max-parallel", [string]$maxParallel,
    "--ports"
) + ($ports | ForEach-Object { [string]$_ }) + @(
    "--stage", $Stage,
    "--ue-exe", $UeExe,
    "--settings-template", $SettingsTemplate,
    "--startup-wait", [string]$StartupWait
)

if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $arguments += @("--resume", $Resume)
}
if ($NoLaunchAirSim) {
    $arguments += "--no-launch-airsim"
}
if ($Execute) {
    $arguments += "--execute"
}

$mode = if ($Execute) { "EXECUTE" } else { "DRY-RUN" }
Write-Host "[$mode] Tunnel mixed checkpoint selection and evaluation"
Write-Host "  Methods: SAC, CoordAtt-SAC"
Write-Host "  Seeds: 0, 1, 2"
Write-Host "  Candidates: best + final + two >=30k-separated return_ma20 peaks + success_rate_ma50 peak"
Write-Host "  Selection: screen only; formal test cannot change the frozen winner"
Write-Host "  Parallel environments: $maxParallel"
Write-Host "  Ports: $($ports -join ', ')"
Write-Host "  Stage: $Stage"
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    Write-Host "  Resume: $Resume"
}

Push-Location -LiteralPath $repoRoot
try {
    & $pixiCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Tunnel mixed evaluation launcher failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (-not $Execute) {
    Write-Host "Dry-run only. After checking the frozen plan, start all stages with:"
    Write-Host "  .\scripts\evaluation\evaluate_env4_local.ps1 -Execute"
}
