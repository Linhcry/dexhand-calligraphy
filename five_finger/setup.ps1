param([string]$UvPath='')
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_LINK_MODE = 'copy'
Push-Location $PSScriptRoot
try {
    if (!$UvPath) {
        $taskUvCommand = Get-Command uv -ErrorAction SilentlyContinue
        if ($taskUvCommand) { $UvPath = $taskUvCommand.Source }
        else { throw 'uv was not found. Install uv from https://docs.astral.sh/uv/ or pass -UvPath.' }
    }
    if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $UvPath venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
    }
    $taskModelSource = Get-Content -LiteralPath 'config\model_sources.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskVendorPath = 'vendor\dexrobot_mujoco'
    if (!(Test-Path -LiteralPath $taskVendorPath)) {
        New-Item -ItemType Directory -Path vendor -Force | Out-Null
        git clone --depth 1 $taskModelSource.repository $taskVendorPath
        if ($LASTEXITCODE -ne 0) { throw 'Official model download failed.' }
        $taskDownloadedRevision = git -C $taskVendorPath rev-parse HEAD
        if ($taskDownloadedRevision -ne $taskModelSource.commit) {
            git -C $taskVendorPath fetch --depth 1 origin $taskModelSource.commit
            if ($LASTEXITCODE -ne 0) { throw 'Pinned model revision download failed.' }
            git -C $taskVendorPath checkout --detach $taskModelSource.commit
            if ($LASTEXITCODE -ne 0) { throw 'Pinned model revision checkout failed.' }
        }
    }
    $taskCurrentRevision = git -C $taskVendorPath rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $taskCurrentRevision -ne $taskModelSource.commit) {
        throw 'The existing vendor repository differs from the recorded revision. It was preserved.'
    }
    & $UvPath pip install --python '.venv\Scripts\python.exe' -r requirements.lock.txt -e .
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.\.venv\Scripts\python.exe' scripts\fetch_hanzi.py
    if ($LASTEXITCODE -ne 0) { throw 'Stroke data download failed.' }
    & '.\.venv\Scripts\python.exe' -m fivefinger.cli doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
    Write-Host 'Setup completed. Run run.ps1 -Mode viewer to open the simulation.'
} finally { Pop-Location }
