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
        exit 0
    }
}
catch {
    # 服务未启动时继续；真正的启动错误在后续健康检查中报告。
}

$previousHost = $env:QDRANT__SERVICE__HOST
try {
    # 本地实验库只监听回环地址，避免未鉴权端口暴露到局域网。
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
            exit 0
        }
    }
    catch {
        # 等待服务完成初始化。
    }
}

throw "Qdrant did not become ready within 15 seconds."
