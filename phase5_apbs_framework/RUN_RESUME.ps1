param(
  [string]$Model = "D:\hf_models\Qwen3.5-9B",
  [string]$ModelKey = "qwen35_9b",
  [ValidateSet("day1", "day2", "full")]
  [string]$Phase = "day1",
  [string]$DType = "bf16",
  [string]$DeviceMap = "auto",
  [switch]$LoadIn4bit,
  [switch]$DryRun,
  [switch]$NoAdoptLegacyResults
)

$ErrorActionPreference = "Stop"

$ArgsList = @(
  "scripts/resume_apbs_experiment.py",
  "--model", $Model,
  "--model-key", $ModelKey,
  "--phase", $Phase,
  "--dtype", $DType,
  "--device-map", $DeviceMap
)

if ($LoadIn4bit) {
  $ArgsList += "--load-in-4bit"
}
if ($DryRun) {
  $ArgsList += "--dry-run"
}
if ($NoAdoptLegacyResults) {
  $ArgsList += "--no-adopt-legacy-results"
}

python @ArgsList
if ($LASTEXITCODE -ne 0) {
  throw "Resume run failed with exit code $LASTEXITCODE"
}

