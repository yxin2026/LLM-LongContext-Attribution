param(
  [string]$Model = "Qwen/Qwen3.5-9B",
  [string]$LocalDir = "D:\hf_models\Qwen3.5-9B",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "== Download model from ModelScope =="
Write-Host "Model: $Model"
Write-Host "LocalDir: $LocalDir"

& $Python -m pip install -U modelscope
if ($LASTEXITCODE -ne 0) {
  throw "Failed to install modelscope"
}

& $Python scripts/download_modelscope.py --model $Model --local-dir $LocalDir
if ($LASTEXITCODE -ne 0) {
  throw "ModelScope download failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "After download, run:"
Write-Host ".\RUN_DAY1.ps1 -Model `"$LocalDir`" -DType bf16 -LoadIn4bit"

