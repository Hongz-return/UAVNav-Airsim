param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string] $Stage,
    [Parameter(Mandatory)][string[]] $PythonArgs
)

$ErrorActionPreference = 'Stop'
$repo = Join-Path $PSScriptRoot '../..'
Set-Location -LiteralPath $repo
$report = 'reports/ch4_stage3_3d_20261125'
New-Item -ItemType Directory -Path $report -Force | Out-Null

function Get-StageSimulator {
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match '^(UnrealEditor|TrainEnv|Tunnel|AirSim|ShaderCompileWorker)'
    }
}

$before = @(Get-StageSimulator)
if ($before.Count) {
    throw "Simulator processes already running: $($before.ProcessName -join ', ')"
}
try {
    & pixi run python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage exited with code $LASTEXITCODE"
    }
}
finally {
    # Python launchers own their UE process trees. Other processes can start during the stage.
    $after = @(Get-StageSimulator)
    [PSCustomObject]@{
        stage = $Stage
        timestamp = Get-Date -Format o
        cleaned = @()
        remaining = @($after | Select-Object Id,ProcessName)
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $report "process_cleanup_$Stage.json") -Encoding utf8
    Write-Host "[$Stage] simulator processes remaining: $($after.Count)"
    if ($after.Count) { throw "Simulator processes are present after $Stage; see the stage report." }
}
