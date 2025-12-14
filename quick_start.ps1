# Quick Start Script for Legal Text Decoder
# Windows PowerShell

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  LEGAL TEXT DECODER - Quick Start" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
Write-Host "[INFO] Working directory: $ProjectDir" -ForegroundColor Yellow
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Cyan
$pythonVersion = & python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found! Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "      Found: $pythonVersion" -ForegroundColor Green

$versionMatch = $pythonVersion -match '(\d+)\.(\d+)'
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "[ERROR] Python 3.10+ required. Current: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Create virtual environment if needed
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    Write-Host "      Creating new virtual environment..." -ForegroundColor Yellow
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "      Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "      Virtual environment already exists." -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Cyan
$activateScript = Join-Path $ProjectDir "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "      Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Could not find activation script. Continuing anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Install dependencies
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Cyan
Write-Host "      This may take 5-10 minutes on first run..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet
& pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies." -ForegroundColor Red
    exit 1
}
Write-Host "      Dependencies installed!" -ForegroundColor Green
Write-Host ""

# Set Python path
Write-Host "[5/6] Configuring Python path..." -ForegroundColor Cyan
$srcPath = Join-Path $ProjectDir "src"
$env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
Write-Host "      PYTHONPATH set to: $srcPath" -ForegroundColor Green
Write-Host ""

# Run main pipeline
Write-Host "[6/6] Running main pipeline..." -ForegroundColor Cyan
Write-Host "      Expected duration: 30-60 minutes (CPU), 10-20 minutes (GPU)" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

& python main.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
    Write-Host "  SUCCESS! Pipeline completed." -ForegroundColor Green
    Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to:" -ForegroundColor Yellow
    Write-Host "  - Models: $(Join-Path $ProjectDir 'models')" -ForegroundColor Cyan
    Write-Host "  - Logs: $(Join-Path $ProjectDir 'log\run.log')" -ForegroundColor Cyan
    Write-Host "  - Data: $(Join-Path $ProjectDir 'data\processed')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View confusion matrices:" -ForegroundColor Yellow
    Write-Host "  explorer $(Join-Path $ProjectDir 'models\confusion_matrices.png')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Try inference:" -ForegroundColor Yellow
    Write-Host "  python src\a04_inference.py --interactive" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "=" -ForegroundColor Red -NoNewline; Write-Host ("=" * 59) -ForegroundColor Red
    Write-Host "  ERROR! Pipeline failed." -ForegroundColor Red
    Write-Host "=" -ForegroundColor Red -NoNewline; Write-Host ("=" * 59) -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs for details:" -ForegroundColor Yellow
    Write-Host "  Get-Content log\run.log -Tail 50" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}
