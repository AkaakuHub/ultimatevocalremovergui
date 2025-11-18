#!/usr/bin/env pwsh

# Ultimate Vocal Remover GUI Launcher
# This script activates virtual environment and launches UVR

# Set error handling
$ErrorActionPreference = "Stop"

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

# Launch UVR
Write-Host "Launching Ultimate Vocal Remover GUI..." -ForegroundColor Cyan
python UVR.py

# Keep window open on error
if ($LASTEXITCODE -ne 0) {
    Write-Host "UVR exited with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}