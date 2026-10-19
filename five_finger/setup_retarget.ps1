param([string]$UvPath='')
$ErrorActionPreference='Stop'
$env:PYTHONIOENCODING='utf-8'
$env:UV_LINK_MODE='copy'
Push-Location $PSScriptRoot
try {
    if (!$UvPath) {
        $taskUvCommand = Get-Command uv -ErrorAction SilentlyContinue
        if ($taskUvCommand) { $UvPath = $taskUvCommand.Source }
        else { throw 'uv was not found. Install uv from https://docs.astral.sh/uv/ or pass -UvPath.' }
    }
    if (!(Test-Path -LiteralPath 'vendor/dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_jaka_zu7.xml')) {
        throw 'Run setup.ps1 first to prepare the base workcell.'
    }
    if (!(Test-Path -LiteralPath '.venv-retarget/Scripts/python.exe')) {
        & $UvPath venv --python '.venv/Scripts/python.exe' .venv-retarget
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
    }
    & $UvPath pip install --python '.venv-retarget/Scripts/python.exe' --only-binary :all: -r requirements-retarget.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $UvPath pip install --python '.venv-retarget/Scripts/python.exe' torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU Torch installation failed.' }
    $taskSource='vendor/dex-retargeting'
    if (!(Test-Path -LiteralPath $taskSource)) {
        git clone --depth 1 https://github.com/dexsuite/dex-retargeting.git $taskSource
        if ($LASTEXITCODE -ne 0) { throw 'Upstream source download failed.' }
        git -C $taskSource fetch --depth 1 origin 3f56141bc8bd2760d5e452e382937269554ebb21
        if ($LASTEXITCODE -ne 0) { throw 'Pinned source download failed.' }
        git -C $taskSource checkout --detach 3f56141bc8bd2760d5e452e382937269554ebb21
        if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed.' }
    }
    & '.\.venv-retarget\Scripts\python.exe' scripts/vendor_dexpilot.py
    if ($LASTEXITCODE -ne 0) { throw 'Port source verification failed.' }
    & '.\.venv-retarget\Scripts\python.exe' -m pytest tests/test_retargeting.py tests/test_competition_rules.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Retargeting verification failed.' }
    Write-Host 'Ready: .venv-retarget\Scripts\python.exe -m experiments.retarget_grasp'
} finally { Pop-Location }
