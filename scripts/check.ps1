param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python virtual environment was not found at $PythonPath. Run scripts\setup.ps1 first."
}

& $PythonPath -m unittest discover -s tests -v

if (Test-Path -LiteralPath "frontend\package.json") {
    Push-Location frontend
    try {
        npm run build
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "Skipping frontend build; frontend\package.json was not found."
}

Write-Host ""
Write-Host "Production check passed."
