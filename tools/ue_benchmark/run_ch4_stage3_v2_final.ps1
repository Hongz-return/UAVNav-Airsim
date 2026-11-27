param(
    [Parameter(Mandatory)][int] $Step,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string] $Label
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '../..')
$report = 'reports/ch4_stage3_v2_20261125'
$caseTag = if ($Label) { "${Step}_$Label" } else { "$Step" }
$screen = Join-Path $report "screens/$caseTag"
$selection = Get-Content -LiteralPath (Join-Path $screen 'screening.json') -Raw | ConvertFrom-Json
if (!$selection.eligible -or (Get-FileHash -LiteralPath $selection.checkpoint).Hash -ne $selection.checkpoint_sha256) {
    throw "Checkpoint $Step did not pass screening or its identity changed"
}
$out = Join-Path $report "final_$caseTag"
New-Item -ItemType Directory -Path $out -Force | Out-Null
$rail = Join-Path $out 'rail'
New-Item -ItemType Directory -Path $rail -Force | Out-Null
$guard = Join-Path $PSScriptRoot 'run_ch4_stage3_v2_guarded.ps1'
$heldout = Join-Path $screen 'heldout'
$protocol = 'reports/ch4_trainenv_3d_20261125/eval_protocol.json'
$project = 'D:/software/Epic Games/UE/UE_5.4/Env/TrainEnvAirSim/TrainEnvAirSim.uproject'

function Invoke-RailCase($scene) {
    $sceneOut = Join-Path $rail $scene
    if (Test-Path -LiteralPath (Join-Path $sceneOut 'result.json')) { return }
    & $guard -Stage "final_${caseTag}_$scene" -PythonArgs @('-u','-m','tools.ue_benchmark.evaluate_ch4_stage3_rail',
        '--scene',$scene,'--run',$selection.training_run,'--checkpoint-step',"$Step",'--episodes','10',
        '--output',$sceneOut) 2>&1 | Tee-Object -FilePath (Join-Path $out "$scene.log")
}

Invoke-RailCase 't1'
& pixi run python -m tools.ue_benchmark.finalize_ch4_stage3_v2 --rail $rail --output $out --rail-gate-sha256 $selection.checkpoint_sha256 --rail-gate-scenes t1
if ($LASTEXITCODE -ne 0) { throw 'Full T1 audit failed' }
if (!(Get-Content -LiteralPath (Join-Path $out 'final_rail_gate.json') -Raw | ConvertFrom-Json).passed) { return }

Invoke-RailCase 't2'
& pixi run python -m tools.ue_benchmark.finalize_ch4_stage3_v2 --rail $rail --output $out --rail-gate-sha256 $selection.checkpoint_sha256
if ($LASTEXITCODE -ne 0) { throw 'Full T1/T2 audit failed' }
if (!(Get-Content -LiteralPath (Join-Path $out 'final_rail_gate.json') -Raw | ConvertFrom-Json).passed) { return }

if (!(Get-Content -LiteralPath (Join-Path $heldout 'summary.json') -Raw | ConvertFrom-Json).complete_protocol) {
    & $guard -Stage "final_${caseTag}_3d" -PythonArgs @('-u','-m','scripts.evaluation.evaluate_trainenv_3d',
        '--candidate-checkpoint',$selection.checkpoint,'--candidate-step',"$Step",'--protocol',$protocol,
        '--ue-project',$project,'--max-instances','2','--run-dir',$heldout,'--resume') 2>&1 |
        Tee-Object -FilePath (Join-Path $out 'heldout.log')
}
Invoke-RailCase 't3'

& pixi run python -m tools.ue_benchmark.summarize_ch4_stage3 --heldout $heldout --rail $rail --output $out
if ($LASTEXITCODE -ne 0) { throw 'Locked benchmark comparison failed' }
& pixi run python -m tools.ue_benchmark.finalize_ch4_stage3_v2 --heldout $heldout --rail $rail --output $out
if ($LASTEXITCODE -ne 0) { throw 'Full rail gate audit or flight evidence failed' }
