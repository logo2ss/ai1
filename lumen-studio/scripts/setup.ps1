# One-time setup for Lumen Studio on Windows (PowerShell).
#   .\scripts\setup.ps1            CPU / DirectML
#   .\scripts\setup.ps1 -Gpu       NVIDIA CUDA
param([switch]$Gpu, [switch]$DirectML)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel
.\.venv\Scripts\pip install -e "engine[dev]"
if ($Gpu) { .\.venv\Scripts\pip uninstall -y onnxruntime; .\.venv\Scripts\pip install "onnxruntime-gpu>=1.17" }
elseif ($DirectML) { .\.venv\Scripts\pip uninstall -y onnxruntime; .\.venv\Scripts\pip install "onnxruntime-directml>=1.17" }
Push-Location frontend; npm ci; npm run build; Pop-Location
Push-Location desktop; npm ci; Pop-Location
.\.venv\Scripts\python scripts\download_models.py
Write-Host "Setup complete. Start with: cd desktop; npx electron ."
