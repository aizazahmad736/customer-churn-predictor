# ChurnGuard AI - PowerShell Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "          Launching ChurnGuard AI Retention Platform                 " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found on your system PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting Streamlit Dashboard at http://localhost:8501..." -ForegroundColor Green
streamlit run app.py
