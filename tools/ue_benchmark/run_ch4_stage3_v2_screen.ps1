param(
    [Parameter(Mandatory)][string] $TrainingRun,
    [Parameter(Mandatory)][int] $Step,
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string] $Label
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '../..')
$run = (Resolve-Path -LiteralPath $TrainingRun).Path
$checkpoint = Join-Path $run "models/checkpoint_${Step}_steps.zip"
if (!(Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint $checkpoint" }
$report = 'reports/ch4_stage3_v2_20261125'
$caseTag = if ($Label) { "${Step}_$Label" } else { "$Step" }
$out = Join-Path $report "screens/$caseTag"
New-Item -ItemType Directory -Path $out -Force | Out-Null
$guard = Join-Path $PSScriptRoot 'run_ch4_stage3_v2_guarded.ps1'
$project = 'D:/software/Epic Games/UE/UE_5.4/Env/TrainEnvAirSim/TrainEnvAirSim.uproject'
$protocol = 'reports/ch4_trainenv_3d_20261125/eval_protocol.json'

$heldout = Join-Path $out 'heldout'
if (!(Test-Path -LiteralPath (Join-Path $heldout 'summary.json'))) {
    $stage = "screen_${caseTag}_3d"
    $arguments = @('-u','-m','scripts.evaluation.evaluate_trainenv_3d',
        '--candidate-checkpoint',$checkpoint,'--candidate-step',"$Step",'--protocol',$protocol,
        '--ue-project',$project,'--max-instances','2','--run-dir',$heldout,'--episodes-per-family','3')
    if (Test-Path -LiteralPath (Join-Path $heldout 'eval_protocol.json')) {
        $stage = "${stage}_resume"
        $arguments += '--resume'
    }
    & $guard -Stage $stage -PythonArgs $arguments 2>&1 | Tee-Object -FilePath (Join-Path $out 'heldout.log') -Append
}
foreach ($scene in 't1','t2') {
    if (Test-Path -LiteralPath (Join-Path $out "$scene/result.json")) { continue }
    & $guard -Stage "screen_${caseTag}_$scene" -PythonArgs @('-u','-m','tools.ue_benchmark.evaluate_ch4_stage3_rail',
        '--scene',$scene,'--run',$run,'--checkpoint-step',"$Step",'--episodes','3',
        '--output',(Join-Path $out $scene)) 2>&1 | Tee-Object -FilePath (Join-Path $out "$scene.log")
}
& pixi run python -m tools.ue_benchmark.screen_ch4_stage3_v2 --screen $out --run $run --step $Step
if ($LASTEXITCODE -ne 0) { throw "Candidate screening failed for step $Step" }
