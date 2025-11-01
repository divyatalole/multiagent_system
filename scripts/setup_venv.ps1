<#
StartupAI - Windows Environment Bootstrap
Creates a Python 3.11 virtual environment and installs project requirements.

Usage (PowerShell):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\scripts\setup_venv.ps1
#>

param(
  [string]$PythonVersion = "3.11",
  [string]$VenvPath = ".venv"
)

Write-Host "==> Checking Python $PythonVersion availability via py launcher..." -ForegroundColor Cyan

$pyCmd = "py -$PythonVersion -c \"import sys; print(sys.version)\""

try {
  $versionOut = & cmd /c $pyCmd 2>$null
} catch {}

if (-not $versionOut) {
  Write-Error "Python $PythonVersion not found. Please install Python $PythonVersion from https://www.python.org/downloads/ and re-run."
  exit 1
}

Write-Host "==> Creating virtual environment at $VenvPath ..." -ForegroundColor Cyan
& cmd /c "py -$PythonVersion -m venv $VenvPath"

if (-not (Test-Path "$VenvPath\Scripts\Activate.ps1")) {
  Write-Error "Failed to create virtual environment at $VenvPath"
  exit 1
}

Write-Host "==> Activating virtual environment..." -ForegroundColor Cyan
& "$VenvPath\Scripts\Activate.ps1"

Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "==> Installing project requirements (this may take a while)..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "\n✅ Environment ready. Next steps:" -ForegroundColor Green
Write-Host "  1) Start API:    .\\$VenvPath\\Scripts\\Activate.ps1; python server_simple.py"
Write-Host "  2) Start UI:     .\\$VenvPath\\Scripts\\Activate.ps1; streamlit run streamlit_app.py"
Write-Host ""
Write-Host "Tip: To deactivate the environment, run 'deactivate'." -ForegroundColor DarkGray


