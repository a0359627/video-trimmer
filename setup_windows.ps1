<#
.SYNOPSIS
    Prepare a Windows machine to run Video Trimmer (local tools only).

.DESCRIPTION
    This script installs and checks local dependencies. It does NOT create or
    change any Google Cloud resource. Use it when an administrator already
    granted your account access to an existing project and bucket.

    Steps:
      1. Check Python 3.10+.
      2. Install FFmpeg and Google Cloud SDK with winget if they are missing.
      3. Create .venv and install Python dependencies.
      4. Write .env when -Project and -Bucket are given and .env does not exist.
      5. Show the ADC login commands that you must run yourself.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Project my-project-id -Bucket my-staging-bucket
#>
param(
    [string]$Project = "",
    [string]$Bucket = "",
    [string]$Location = "global",
    [string]$Model = "gemini-3.8-flash"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "[*] $msg" }
function Write-Ok($msg) { Write-Host "    [OK] $msg" }

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

# 1. Python
Write-Step "Checking Python 3.10+..."
$python = $null
foreach ($candidate in @("py", "python")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $ver = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($ver -and [version]$ver -ge [version]"3.10") { $python = $candidate; break }
    }
}
if (-not $python) {
    throw "Python 3.10 or later is not found. Install it from https://www.python.org/downloads/ and run this script again."
}
Write-Ok "Python $ver ($python)"

# 2. FFmpeg and Google Cloud SDK
foreach ($tool in @(
    @{ Cmd = "ffmpeg"; Id = "Gyan.FFmpeg"; Name = "FFmpeg" },
    @{ Cmd = "gcloud"; Id = "Google.CloudSDK"; Name = "Google Cloud SDK" }
)) {
    Write-Step "Checking $($tool.Name)..."
    if (-not (Get-Command $tool.Cmd -ErrorAction SilentlyContinue)) {
        # A tool installed after this terminal opened is not on the session PATH yet.
        Update-SessionPath
    }
    if (-not (Get-Command $tool.Cmd -ErrorAction SilentlyContinue)) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "$($tool.Name) is missing and winget is not available. Install $($tool.Name) manually."
        }
        winget install --id $tool.Id -e --silent --accept-source-agreements --accept-package-agreements
        Update-SessionPath
    }
    if (-not (Get-Command $tool.Cmd -ErrorAction SilentlyContinue)) {
        throw "$($tool.Name) is installed but not on PATH. Open a new terminal and run this script again."
    }
    Write-Ok "$($tool.Name) found"
}

# 3. Virtual environment and Python dependencies
Write-Step "Creating .venv and installing Python dependencies..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }
Write-Ok "Dependencies installed in .venv"

# 4. .env
Write-Step "Checking .env..."
if (Test-Path ".env") {
    Write-Ok ".env already exists (not changed)"
} elseif ($Project -and $Bucket) {
    $lines = @(
        "GOOGLE_CLOUD_PROJECT=$Project",
        "GOOGLE_CLOUD_LOCATION=$Location",
        "VIDEO_TRIMMER_BUCKET=$Bucket",
        "MODEL_NAME=$Model"
    )
    [System.IO.File]::WriteAllLines((Join-Path $RepoRoot ".env"), $lines)
    Write-Ok ".env written"
} else {
    Write-Host "    [!] .env is missing. Run again with -Project and -Bucket, or copy .env.example to .env and edit it."
}

# 5. ADC login (interactive, run by the user)
$adcProject = if ($Project) { $Project } else { "YOUR_PROJECT_ID" }
Write-Host ""
Write-Host "Next steps (run these yourself and sign in with your authorized account):"
Write-Host "  gcloud auth application-default login --project=$adcProject"
Write-Host "  gcloud auth application-default set-quota-project $adcProject"
Write-Host ""
Write-Host "Then run Video Trimmer:"
Write-Host "  `$env:PYTHONUTF8 = '1'"
Write-Host "  .\.venv\Scripts\python.exe skills\video-trimmer\scripts\video_trimmer.py -i `"C:\path\to\raw_footage.mp4`""
