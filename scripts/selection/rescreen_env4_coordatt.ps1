[CmdletBinding()]
param(
    [switch]$Execute,
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
    "scripts.selection.select_env4_coordatt",
    "--training-root", $TrainingRoot,
    "--output-root", $OutputRoot,
    "--max-parallel", [string]$maxParallel,
    "--ports"
) + ($ports | ForEach-Object { [string]$_ }) + @(
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
Write-Host "[$mode] ENV4 CoordAtt-only checkpoint rescreen"
Write-Host "  Methods: CoordAtt-SAC only"
Write-Host "  Seeds: 0, 1, 2"
Write-Host "  Candidates: every saved 10k periodic checkpoint + best/final aliases"
Write-Host "  Validation: two independent 10-episode batches (scenario seeds 33001/33002)"
Write-Host "  Gate: all-zero candidate groups become NO_VALID_CHECKPOINT"
Write-Host "  Formal test: disabled in this launcher"
Write-Host "  Parallel environments: $maxParallel"
Write-Host "  Ports: $($ports -join ', ')"
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    Write-Host "  Resume: $Resume"
}

Push-Location -LiteralPath $repoRoot
try {
    & $pixiCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "ENV4 CoordAtt rescreen launcher failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (-not $Execute) {
    Write-Host "Dry-run only. After checking the frozen plan, start screening with:"
    Write-Host "  .\scripts\selection\rescreen_env4_coordatt.ps1 -Execute"
}
