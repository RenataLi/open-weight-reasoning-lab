param(
    [ValidateSet("search", "confirm")]
    [string]$Phase = "search",
    [string]$Config = "configs/stage4_dpo_qwen3_1_7b.yaml",
    [string]$SearchRun = "",
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TQDM_DISABLE = "1"
$env:PYTHONUNBUFFERED = "1"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

if ($Phase -eq "search") {
    $arguments = @("-m", "reasoning_lab.cli", "run-stage4-search", "--config", $Config)
} else {
    if ([string]::IsNullOrWhiteSpace($SearchRun)) {
        throw "-SearchRun is required for the confirm phase."
    }
    $arguments = @(
        "-m", "reasoning_lab.cli", "run-stage4-confirm", "--config", $Config,
        "--search-run", $SearchRun
    )
}
if (-not [string]::IsNullOrWhiteSpace($RunDir)) {
    $arguments += @("--run-dir", $RunDir)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
