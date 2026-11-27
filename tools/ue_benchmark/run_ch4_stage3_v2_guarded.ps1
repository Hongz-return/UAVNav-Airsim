param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string] $Stage,
    [Parameter(Mandatory)][string[]] $PythonArgs
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '../..')
$report = 'reports/ch4_stage3_v2_20261125'

function Get-StageSimulator {
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match '^(UnrealEditor|TrainEnv|Tunnel|AirSim|ShaderCompileWorker)'
    }
}

$before = @(Get-StageSimulator)
if ($before.Count) {
    throw "Simulator preflight found an active process: $($before.ProcessName -join ', ')"
}

try {
    & pixi run python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage exited with code $LASTEXITCODE"
    }
} finally {
    # The launcher also owns a Windows kill-on-close Job for its UE process trees.
    $remainder = @(Get-StageSimulator)
    foreach ($process in $remainder) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $process.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
    $after = @(Get-StageSimulator)
    [pscustomobject]@{
        stage = $Stage
        timestamp = Get-Date -Format o
        cleaned = @($remainder | Select-Object Id,ProcessName)
        remaining = @($after | Select-Object Id,ProcessName)
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $report "process_cleanup_$Stage.json") -Encoding utf8
    Write-Host "[$Stage] simulator processes remaining: $($after.Count)"
    if ($after.Count) { throw "$Stage left simulator processes" }
}
