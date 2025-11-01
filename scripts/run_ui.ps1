<#
Run the Streamlit web UI inside the local virtual environment.
Usage:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\run_ui.ps1
#>

param(
  [string]$VenvPath = ".venv",
  [int]$Port = 8501
)

if (-not (Test-Path "$VenvPath\Scripts\Activate.ps1")) {
  Write-Error "Virtual environment not found at $VenvPath. Run scripts\\setup_venv.ps1 first."
  exit 1
}

& "$VenvPath\Scripts\Activate.ps1"

Write-Host "==> Starting Streamlit UI on http://localhost:$Port ..." -ForegroundColor Cyan
streamlit run streamlit_app.py --server.port $Port


