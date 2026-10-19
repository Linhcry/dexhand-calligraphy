param(
    [ValidateSet('viewer','demo','doctor','plan','ocr','prepare','ik-check','check')][string]$Mode='viewer',
    [string]$Text='永'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first.' }
    if ($Mode -eq 'check') { & $taskPython -m pytest -q }
    elseif ($Mode -eq 'plan') { & $taskPython -m fivefinger.cli plan --text $Text }
    elseif ($Mode -eq 'demo') { & $taskPython -m fivefinger.cli demo --headless --seconds 6 }
    else { & $taskPython -m fivefinger.cli $Mode }
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
