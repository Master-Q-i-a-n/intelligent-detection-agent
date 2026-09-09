param(
    [string]$QdrantDir = "D:\Qdrant"
)

$ErrorActionPreference = "Stop"
$qdrantExe = Join-Path $QdrantDir "qdrant.exe"
$readyUrl = "http://127.0.0.1:6333/readyz"

if (-not (Test-Path -LiteralPath $qdrantExe -PathType Leaf)) {
    throw "Qdrant executable not found: $qdrantExe"
}

try {
    $response = Invoke-WebRequest -Uri $readyUrl -TimeoutSec 2 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "Qdrant is already ready: $readyUrl" -ForegroundColor Green
        # Return to the caller without terminating the parent PowerShell host.
        return
    }
}
catch {
    # Continue when the service is offline; startup checks report real failures.
}

$previousHost = $env:QDRANT__SERVICE__HOST
try {
    # Bind the unauthenticated local database to loopback only.
    $env:QDRANT__SERVICE__HOST = "127.0.0.1"
    $process = Start-Process `
        -FilePath $qdrantExe `
        -WorkingDirectory $QdrantDir `
        -ArgumentList @("--disable-telemetry") `
        -PassThru `
        -WindowStyle Hidden
}
finally {
    $env:QDRANT__SERVICE__HOST = $previousHost
}

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        throw "Qdrant exited during startup with code $($process.ExitCode)."
    }
    try {
        $response = Invoke-WebRequest -Uri $readyUrl -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "Qdrant ready: $readyUrl (PID $($process.Id))" -ForegroundColor Green
            return
        }
    }
    catch {
        # Wait for Qdrant initialization to finish.
    }
}

throw "Qdrant did not become ready within 15 seconds."
