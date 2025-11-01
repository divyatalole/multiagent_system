<#
Run the FastAPI backend inside the local virtual environment.
Usage:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\run_api.ps1
#>

param(
  [string]$VenvPath = ".venv",
  [int]$Port = 8000
)

if (-not (Test-Path "$VenvPath\Scripts\Activate.ps1")) {
  Write-Error "Virtual environment not found at $VenvPath. Run scripts\\setup_venv.ps1 first."
  exit 1
}

& "$VenvPath\Scripts\Activate.ps1"

Write-Host "==> Starting API server on http://localhost:$Port ..." -ForegroundColor Cyan
python server_simple.py


