param([string]$InputTextPath, [string]$OutputWavePath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer
$chinese = $speech.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'zh-CN' } | Select-Object -First 1
if (-not $chinese) { throw 'No installed Chinese speech voice' }
$speech.SelectVoice($chinese.VoiceInfo.Name)
$speech.Rate = 0
$speech.SetOutputToWaveFile($OutputWavePath)
$speech.Speak([System.IO.File]::ReadAllText($InputTextPath, [System.Text.Encoding]::UTF8))
$speech.Dispose()
