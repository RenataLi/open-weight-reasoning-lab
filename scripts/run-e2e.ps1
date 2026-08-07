param(
    [Parameter(Mandatory = $false)]
    [string]$Config = "configs\e2e_smoke.yaml",

    [Parameter(Mandatory = $false)]
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python"
}

$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"

Set-Location -LiteralPath $projectRoot

$arguments = @("-m", "reasoning_lab.cli", "run", "--config", $Config)
if ($RunDir) {
    $arguments += @("--run-dir", $RunDir)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "The end-to-end pipeline failed with exit code $LASTEXITCODE"
}

