param(
  [string]$Model = "Qwen/Qwen3.5-9B",
  [string]$LocalDir = "D:\hf_models\Qwen3.5-9B",
  [string]$Endpoint = "https://hf-mirror.com",
  [string]$Python = "python",
  [switch]$Snapshot
)

$ErrorActionPreference = "Stop"

Write-Host "== Download model first, then run experiments from local path =="
Write-Host "Model: $Model"
Write-Host "LocalDir: $LocalDir"
Write-Host "Endpoint: $Endpoint"

$DirectArg = "--direct"
if ($Snapshot) {
  $DirectArg = ""
}

& $Python scripts/download_model.py --model $Model --local-dir $LocalDir --endpoint $Endpoint $DirectArg
if ($LASTEXITCODE -ne 0) {
  throw "Model download failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "After download, run:"
Write-Host ".\RUN_DAY1.ps1 -Model `"$LocalDir`" -DType bf16 -LoadIn4bit"
