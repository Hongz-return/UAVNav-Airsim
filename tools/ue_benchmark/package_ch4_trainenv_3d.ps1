param(
    [string]$EngineRoot = 'D:/software/Epic Games/UE/UE_5.4',
    [string]$Project = 'D:/software/Epic Games/UE/UE_5.4/Env/TrainEnvAirSim/TrainEnvAirSim.uproject',
    [string]$Archive = 'D:/UEBuilds/TrainEnv3D'
)

$ErrorActionPreference = 'Stop'
$map = '/Game/UAVTraining/Maps/TrainEnv_3D'
$projectFile = (Resolve-Path -LiteralPath $Project).Path
$archivePath = [IO.Path]::GetFullPath($Archive)
$oldArchivePath = [IO.Path]::GetFullPath('D:/UEBuilds/TrainEnv')
if ($archivePath.TrimEnd('\') -eq $oldArchivePath -or $archivePath.StartsWith("$oldArchivePath\",
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Choose an archive directory separate from the original TrainEnv deployment."
}
$mapFile = Join-Path (Split-Path -Parent $projectFile) 'Content/UAVTraining/Maps/TrainEnv_3D.umap'
if (-not (Test-Path -LiteralPath $mapFile -PathType Leaf)) {
    throw "Generate $map before packaging."
}
$uat = Join-Path $EngineRoot 'Engine/Build/BatchFiles/RunUAT.bat'
& $uat BuildCookRun "-project=$projectFile" -noP4 -unattended -utf8output -platform=Win64 `
    -clientconfig=Development -skipbuild -cook "-map=$map" -stage -pak -iostore -package -archive `
    "-stagingdirectory=$(Join-Path $archivePath 'Stage')" "-archivedirectory=$archivePath"
if ($LASTEXITCODE -ne 0) {
    throw "TrainEnv_3D packaging failed with exit code $LASTEXITCODE."
}
