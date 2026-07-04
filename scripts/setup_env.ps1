# DRW 项目环境配置脚本
# 用法: 在项目根目录执行 .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PyLauncher = "C:\Users\31304\AppData\Local\Programs\Python\Python313\python.exe"

Write-Host "项目目录: $ProjectRoot"

if (-not (Test-Path $VenvPython)) {
    Write-Host "创建虚拟环境..."
    if (-not (Test-Path $PyLauncher)) {
        throw "未找到 Python 3.13: $PyLauncher"
    }
    & $PyLauncher -m venv (Join-Path $ProjectRoot ".venv")
}

Write-Host "安装依赖..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "运行环境校验..."
& $VenvPython (Join-Path $ProjectRoot "src\check_env.py")
if ($LASTEXITCODE -ne 0) {
    throw "环境校验失败"
}

Write-Host ""
Write-Host "环境配置完成。"
Write-Host "激活虚拟环境:  $ProjectRoot\.venv\Scripts\Activate.ps1"
Write-Host "下载数据:      .\scripts\download_data.ps1"
