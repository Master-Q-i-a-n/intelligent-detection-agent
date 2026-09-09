param(
    # Production builds are served by FastAPI, so Vite can be skipped.
    [switch]$ApiOnly,
    # Keep the local vector database location configurable for other machines.
    [string]$QdrantDir = "D:\Qdrant"
)

$ErrorActionPreference = "Stop"

# Always run from the project root so relative paths remain stable.
Push-Location $PSScriptRoot
$frontendProcess = $null
$frontendOutput = $null
$frontendError = $null
try {
    # RAG is an API dependency. The helper is idempotent and waits for readyz.
    & (Join-Path $PSScriptRoot "scripts\start_qdrant.ps1") `
        -QdrantDir $QdrantDir

    if (-not $ApiOnly) {
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if ($null -eq $npmCommand) {
            throw "npm.cmd was not found. Install Node.js and reopen the terminal."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "frontend\node_modules"))) {
            throw "Frontend dependencies are missing. Run: npm --prefix frontend install"
        }

        # Keep each run separate if a previous process still holds its log files.
        $logPrefix = "yaoheng-vite-$PID-$([guid]::NewGuid().ToString('N'))"
        $frontendOutput = Join-Path ([IO.Path]::GetTempPath()) "$logPrefix.out.log"
        $frontendError = Join-Path ([IO.Path]::GetTempPath()) "$logPrefix.err.log"
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

    uv run python -m intelligent_detection_agent
}
finally {
    try {
        if ($null -ne $frontendProcess) {
            try {
                if (-not $frontendProcess.HasExited) {
                    # Wait for tree termination before attempting log cleanup.
                    $stopProcess = Start-Process -FilePath taskkill.exe `
                        -ArgumentList @("/PID", $frontendProcess.Id, "/T", "/F") `
                        -WindowStyle Hidden -PassThru
                    try {
                        if (-not $stopProcess.WaitForExit(5000)) {
                            Write-Warning "Vite process-tree shutdown is still pending."
                        }
                    }
                    finally {
                        $stopProcess.Dispose()
                    }
                    if (-not $frontendProcess.WaitForExit(5000)) {
                        Write-Warning "Vite has not exited yet (PID $($frontendProcess.Id))."
                    }
                }
            }
            catch {
                # Cleanup failures must not replace the original startup/runtime error.
                Write-Warning "Could not finish Vite shutdown: $($_.Exception.Message)"
            }
            finally {
                $frontendProcess.Dispose()
            }
        }
        foreach ($logPath in @($frontendOutput, $frontendError)) {
            if ($null -eq $logPath) { continue }
            # Redirected output handles may take a moment to close after process exit.
            for ($attempt = 0; $attempt -lt 5; $attempt++) {
                try {
                    if (Test-Path -LiteralPath $logPath) {
                        Remove-Item -LiteralPath $logPath -Force -ErrorAction Stop
                    }
                    break
                }
                catch {
                    if ($attempt -eq 4) {
                        Write-Warning "Temporary log retained: $logPath ($($_.Exception.Message))"
                    }
                    else {
                        Start-Sleep -Milliseconds 200
                    }
                }
            }
        }
    }
    finally {
        Pop-Location
    }
}
