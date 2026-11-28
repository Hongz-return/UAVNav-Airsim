param(
    [string]$EngineRoot = 'D:/software/Epic Games/UE/UE_5.9',
    [string]$Project = 'D:/UEProjects/Ch4Tunnel_UE59/Ch4Tunnel.uproject',
    [string]$Archive = 'D:/UEBuilds/NavRLForest_UE59',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
# Prevent .NET UAT from routing local ZenServer ([::1]:8558) through an HTTP/SOCKS proxy.
foreach ($proxyVar in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')) {
    Remove-Item "Env:$proxyVar" -ErrorAction SilentlyContinue
}
$autoSdk = 'D:\UEProjects\Toolchains\WindowsSDK-26100\AutoSDK'
if ((Test-Path -LiteralPath $autoSdk) -and -not $env:UE_SDKS_ROOT) {
    $env:UE_SDKS_ROOT = $autoSdk
}
$maps = @(
    '/Game/Training/Maps/L_Train_NavRL_Forest_V3_Stage1',
    '/Game/Training/Maps/L_Train_NavRL_Forest_V3_Stage2',
    '/Game/Training/Maps/L_Train_NavRL_Forest_V3_Stage3'
)
$projectFile = (Resolve-Path -LiteralPath $Project).Path
$projectDir = Split-Path -Parent $projectFile
foreach ($map in $maps) {
    $rel = $map -replace '^/Game/', ''
    $mapFile = Join-Path $projectDir ("Content/" + $rel + ".umap")
    if (-not (Test-Path -LiteralPath $mapFile -PathType Leaf)) {
        throw "Missing map asset before packaging: $mapFile"
    }
}
$archivePath = [IO.Path]::GetFullPath($Archive)
New-Item -ItemType Directory -Force -Path $archivePath | Out-Null
$uat = Join-Path $EngineRoot 'Engine/Build/BatchFiles/RunUAT.bat'
$mapArg = $maps -join '+'
$buildFlag = if ($SkipBuild) { '-skipbuild' } else { '-build' }
& $uat BuildCookRun "-project=$projectFile" -noP4 -unattended -utf8output -platform=Win64 `
    -clientconfig=Development $buildFlag -cook "-map=$mapArg" -stage -pak -iostore -package -archive `
    "-stagingdirectory=$(Join-Path $archivePath 'Stage')" "-archivedirectory=$archivePath"
if ($LASTEXITCODE -ne 0) {
    throw "NavRL Forest UE5.9 packaging failed with exit code $LASTEXITCODE."
}
