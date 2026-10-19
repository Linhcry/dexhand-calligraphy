$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$packetRoot = Join-Path $projectRoot 'deliverables/报名准备_20260910'
$qaRoot = Join-Path $projectRoot 'outputs/registration_doc_qa'
New-Item -ItemType Directory -Force -Path $qaRoot | Out-Null
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    foreach ($label in @('技术资格仿真说明', '队伍信息采集表')) {
        $sourceFile = Join-Path $packetRoot ($label + '.docx')
        $pdfFile = Join-Path $qaRoot ($label + '.pdf')
        $document = $wordApp.Documents.Open($sourceFile, $false, $true)
        try { $document.ExportAsFixedFormat($pdfFile, 17) }
        finally { $document.Close(0) }
        Write-Output $pdfFile
    }
} finally { $wordApp.Quit() }
