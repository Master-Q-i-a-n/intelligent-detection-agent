param(
    # Production builds are served by FastAPI, so Vite can be skipped.
    [switch]$ApiOnly
)

$ErrorActionPreference = "Stop"

# Always run from the project root so relative paths remain stable.
Push-Location $PSScriptRoot
$frontendProcess = $null
$frontendOutput = $null
$frontendError = $null
try {
    if (-not $ApiOnly) {
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if ($null -eq $npmCommand) {
            throw "npm.cmd was not found. Install Node.js and reopen the terminal."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "frontend\node_modules"))) {
            throw "Frontend dependencies are missing. Run: npm --prefix frontend install"
        }

        $frontendOutput = Join-Path ([IO.Path]::GetTempPath()) "yaoheng-vite-$PID.out.log"
        $frontendError = Join-Path ([IO.Path]::GetTempPath()) "yaoheng-vite-$PID.err.log"
        $frontendProcess = Start-Process `
            -FilePath $npmCommand.Source `
            -ArgumentList @(
                "--prefix", "frontend", "run", "dev", "--",
                "--host", "127.0.0.1", "--port", "5173", "--strictPort"
            ) `
            -WorkingDirectory $PSScriptRoot `
            -PassThru `
            -WindowStyle Hidden `
            -RedirectStandardOutput $frontendOutput `
            -RedirectStandardError $frontendError

        Start-Sleep -Milliseconds 1200
        if ($frontendProcess.HasExited) {
            $details = @(
                Get-Content -LiteralPath $frontendOutput -Encoding UTF8 -ErrorAction SilentlyContinue
                Get-Content -LiteralPath $frontendError -Encoding UTF8 -ErrorAction SilentlyContinue
            ) -join [Environment]::NewLine
            throw "Vite failed to start.$([Environment]::NewLine)$details"
        }
        Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Cyan
        Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop both services." -ForegroundColor DarkGray
    }

    uv run python .\run_api.py
}
finally {
    if ($null -ne $frontendProcess -and -not $frontendProcess.HasExited) {
        # npm spawns Vite/Node children, so terminate the complete process tree.
        & taskkill.exe /PID $frontendProcess.Id /T /F 2>$null | Out-Null
    }
    foreach ($logPath in @($frontendOutput, $frontendError)) {
        if ($null -ne $logPath -and (Test-Path -LiteralPath $logPath)) {
            Remove-Item -LiteralPath $logPath -Force
        }
    }
    Pop-Location
}
