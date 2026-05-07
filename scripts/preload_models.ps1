param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $VenvPath "Scripts\python.exe"

@'
from app.settings import Settings
from app.inference import LocalInferenceService

settings = Settings.from_env()
service = LocalInferenceService(settings)
service._ensure_runtime()
print("Model preload completed.")
'@ | & $python -
