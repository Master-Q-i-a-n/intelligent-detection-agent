$ErrorActionPreference = "Stop"

# 始终从项目根目录启动，避免调用脚本时当前目录不同导致相对路径失效。
Push-Location $PSScriptRoot
try {
    uv run python .\run_api.py
}
finally {
    Pop-Location
}
