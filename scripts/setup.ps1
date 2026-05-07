param(
    [string]$VenvPath = ".venv",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$TorchVersion = "2.9.1",
    [string]$TorchVisionVersion = "0.24.1",
    [string]$TorchAudioVersion = "2.9.1"
)

$ErrorActionPreference = "Stop"

py -3.11 -m venv $VenvPath
$python = Join-Path $VenvPath "Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install --force-reinstall `
    "torch==$TorchVersion" `
    "torchvision==$TorchVisionVersion" `
    "torchaudio==$TorchAudioVersion" `
    --index-url $TorchIndexUrl
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\$VenvPath\Scripts\Activate.ps1"
Write-Host "Run with:      python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
