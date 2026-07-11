# ARGUS Launcher - ASCII only, no Unicode box chars
# Usage: .\launch_argus.ps1 [-Demo] [-SkipInstall] [-TrainFirst]

param(
    [switch]$Demo        = $false,
    [switch]$SkipInstall = $false,
    [switch]$TrainFirst  = $false,
    [switch]$NoDisplay   = $false,
    [string]$VideoSource = "synthetic"
)

$ErrorActionPreference = "Stop"

function Write-Cyan   { param($msg) Write-Host $msg -ForegroundColor Cyan    }
function Write-Green  { param($msg) Write-Host $msg -ForegroundColor Green   }
function Write-Red    { param($msg) Write-Host $msg -ForegroundColor Red     }
function Write-Yellow { param($msg) Write-Host $msg -ForegroundColor Yellow  }

$ARGUS_ROOT = Split-Path $MyInvocation.MyCommand.Path -Parent
$ML_ENGINE  = Join-Path $ARGUS_ROOT "ml_engine"
$AGENTIC    = Join-Path $ARGUS_ROOT "agentic_engine"
$DASH_DIR   = Join-Path $AGENTIC    "dashboard"

Write-Host ""
Write-Cyan  "+=========================================================+"
Write-Cyan  "|          PROJECT ARGUS - SYSTEM LAUNCHER               |"
Write-Cyan  "|    Predictive Threat Detection System v1.0              |"
Write-Cyan  "|    Lead Architect: Abdullah Javed (vagabond)            |"
Write-Cyan  "+=========================================================+"
Write-Host ""

# --- Step 1: Python Check ---
Write-Yellow "[1/6] Checking Python..."
try {
    $pyVer = python --version 2>&1
    Write-Green "  OK: $pyVer"
} catch {
    Write-Red "  FAIL: Python not found. Install from https://python.org"
    exit 1
}

# --- Step 2: Install Dependencies ---
if ($SkipInstall) {
    Write-Yellow "[2/6] Skipping dependency install (SkipInstall flag is set)"
} else {
    Write-Yellow "[2/6] Installing ML Engine dependencies..."
    Push-Location $ML_ENGINE
    pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) {
        Write-Red "  FAIL: ML deps install failed"
        Pop-Location
        exit 1
    }
    Write-Green "  OK: ML Engine deps installed"
    Pop-Location

    Write-Yellow "[2/6] Installing Agentic Engine dependencies..."
    Push-Location $AGENTIC
    pip install -r requirements.txt -q
    if ($LASTEXITCODE -ne 0) {
        Write-Red "  FAIL: Agentic deps install failed"
        Pop-Location
        exit 1
    }
    Write-Green "  OK: Agentic Engine deps installed"
    Pop-Location
}

# --- Step 3: Generate Synthetic Data ---
Write-Yellow "[3/6] Generating synthetic test data..."
Push-Location $ML_ENGINE
python data\synthetic.py 2>&1 | Out-Null
Pop-Location
Write-Green "  OK: Synthetic data ready"

# --- Step 3b: Train Models (optional) ---
if ($TrainFirst) {
    Write-Yellow "[3b]  Training models on synthetic data..."
    Push-Location $ML_ENGINE
    python train_all.py --synthetic --epochs 20
    Pop-Location
    Write-Green "  OK: Models trained"
}

# --- Step 4: Start FastAPI Sensor Hub ---
Write-Yellow "[4/6] Starting FastAPI Sensor Hub on port 8001..."
$sensorJob = Start-Job -Name "ARGUS-SensorHub" -ScriptBlock {
    param($apiDir)
    Set-Location $apiDir
    python sensor_hub.py
} -ArgumentList (Join-Path $ML_ENGINE "api")

Start-Sleep -Seconds 4

try {
    $resp = Invoke-RestMethod -Uri "http://localhost:8001/status" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Green "  OK: Sensor Hub is online"
} catch {
    Write-Yellow "  NOTE: Sensor Hub may still be starting up"
}

# --- Step 5: Start Dashboard ---
Write-Yellow "[5/6] Starting ARGUS Dashboard on port 5000..."
$dashJob = Start-Job -Name "ARGUS-Dashboard" -ScriptBlock {
    param($dir)
    Set-Location $dir
    python app.py
} -ArgumentList $DASH_DIR

Start-Sleep -Seconds 5
Write-Green "  OK: Dashboard launching..."

# --- Open Browser ---
Write-Yellow "[5b]  Opening browser..."
try {
    Start-Process "http://localhost:5000"
    Write-Green "  OK: Browser opened -> http://localhost:5000"
} catch {
    Write-Yellow "  NOTE: Open manually -> http://localhost:5000"
}

# --- Step 6: Start ML Pipeline ---
Write-Yellow "[6/6] Starting ML Vision Pipeline..."
Write-Host ""

$src = if ($Demo) { "synthetic" } else { $VideoSource }
$argsList = "--input $src"
if ($NoDisplay) { $argsList += " --no-display" }

Write-Cyan  "+========================================================+"
Write-Cyan  "  ARGUS IS ACTIVE"
Write-Cyan  "  Dashboard  : http://localhost:5000"
Write-Cyan  "  Sensor Hub : http://localhost:8001/docs"
Write-Cyan  "  Video src  : $src"
Write-Cyan  "  Press Ctrl+C to stop"
Write-Cyan  "+========================================================+"
Write-Host ""

Push-Location $ML_ENGINE
try {
    Invoke-Expression "python models\pipeline.py $argsList"
    Write-Cyan "`nVideo ended. Pipeline stopped, but Dashboard is still active."
    Write-Cyan "Press Ctrl+C to terminate the system."
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Write-Yellow "`nShutting down background services..."
    Stop-Job  -Name "ARGUS-SensorHub" -ErrorAction SilentlyContinue
    Stop-Job  -Name "ARGUS-Dashboard"  -ErrorAction SilentlyContinue
    Remove-Job -Name "ARGUS-SensorHub" -ErrorAction SilentlyContinue
    Remove-Job -Name "ARGUS-Dashboard"  -ErrorAction SilentlyContinue
    Write-Green "ARGUS shutdown complete."
}
Pop-Location
