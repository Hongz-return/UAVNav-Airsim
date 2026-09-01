[CmdletBinding()]
param(
    [switch]$Execute,
    [ValidateRange(1, 2147483647)]
    [int]$Timesteps = 200000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ueExe = "D:\UEBuilds\TunnelEnv\Windows\Tunnel.exe"
$settingsTemplate = "C:\Users\31732\Documents\AirSim\settings.json"
$mapPath = "/Game/Tunnel_Col_Vol1/Maps/Tunnel_AO"
$methods = @("sac", "coordatt", "lstm")
$seeds = @(0, 1, 2)
$ports = @(41451, 41461, 41471)

$pixiCommand = Get-Command pixi -ErrorAction Stop

if ($Execute) {
    if (-not (Test-Path -LiteralPath $ueExe -PathType Leaf)) {
        throw "Tunnel UE executable not found: $ueExe"
    }
    if (-not (Test-Path -LiteralPath $settingsTemplate -PathType Leaf)) {
        throw "AirSim settings template not found: $settingsTemplate"
    }
}

$arguments = @(
    "run",
    "python",
    "-m",
    "scripts.training.train_parallel",
    "--scene", "tunnel_mixed",
    "--methods"
) + $methods + @(
    "--seeds"
) + ($seeds | ForEach-Object { [string]$_ }) + @(
    "--max-parallel", "3",
    "--ports"
) + ($ports | ForEach-Object { [string]$_ }) + @(
    "--timesteps", [string]$Timesteps,
    "--ue-exe", $ueExe,
    "--settings-template", $settingsTemplate,
    "--clock-speed", "1.0",
    "--res-x", "640",
    "--res-y", "480",
    "--startup-wait", "60",
    "--log-root", "Logs/TrainData/Runs"
)

if (-not $Execute) {
    $arguments += "--dry-run"
}

$mode = if ($Execute) { "EXECUTE" } else { "DRY-RUN" }
Write-Host "[$mode] Mixed tunnel training"
Write-Host "  UE: $ueExe"
Write-Host "  Map: $mapPath"
Write-Host "  Methods: $($methods -join ', ')"
Write-Host "  Seeds: $($seeds -join ', ')"
Write-Host "  Timesteps/job: $Timesteps"
Write-Host "  Parallel ports: $($ports -join ', ')"

Push-Location $repoRoot
try {
    & $pixiCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Mixed-scene training launcher failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if (-not $Execute) {
    Write-Host "Dry-run only. Start formal training with:"
    Write-Host "  .\scripts\training\train_env4_local.ps1 -Execute"
}
