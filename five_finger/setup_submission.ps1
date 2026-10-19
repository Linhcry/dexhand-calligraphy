param([string]$Python='python', [switch]$WithRetargeting)
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
Push-Location $PSScriptRoot
try {
    & $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'"
    if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.12: -Python C:\path\python.exe' }
    foreach ($taskEnv in @('.venv','.venv-retarget')) {
        if (!(Test-Path -LiteralPath "$taskEnv/Scripts/python.exe")) {
            & $Python -m venv $taskEnv
            if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
        }
    }
    & ./.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed' }
    & ./.venv-retarget/Scripts/python.exe -m pip install -r requirements-retarget.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Simulation dependency installation failed' }
    if ($WithRetargeting) {
        & ./.venv-retarget/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'Optional retargeting dependency installation failed' }
    }
    & ./.venv-retarget/Scripts/python.exe main.py check
    if ($LASTEXITCODE -ne 0) { throw 'Package check failed' }
    Write-Host 'Ready for local reference simulation. No hardware driver is configured.'
} finally { Pop-Location }
