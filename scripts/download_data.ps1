# DRW 项目数据下载脚本
# 用法:
#   方式 A (推荐): 先执行 kaggle 登录，再运行本脚本
#     .\.venv\Scripts\python.exe -m kaggle auth login
#     .\scripts\download_data.ps1
#   方式 B: 设置环境变量后运行
#     $env:KAGGLE_API_TOKEN = "你的token"
#     .\scripts\download_data.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data\raw"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AccessTokenPath = Join-Path $env:USERPROFILE ".kaggle\access_token"

if (-not (Test-Path $VenvPython)) {
    throw "未找到虚拟环境，请先运行 .\scripts\setup_env.ps1"
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

$hasToken = $false
if ($env:KAGGLE_API_TOKEN) {
    $hasToken = $true
    Write-Host "检测到环境变量 KAGGLE_API_TOKEN"
}
elseif (Test-Path $AccessTokenPath) {
    $hasToken = $true
    Write-Host "检测到 Kaggle 凭证文件: $AccessTokenPath"
}

if (-not $hasToken) {
    Write-Host ""
    Write-Host "尚未配置 Kaggle API 凭证。请先完成以下任一操作:"
    Write-Host "1. 浏览器登录 (推荐):"
    Write-Host "   $VenvPython -m kaggle auth login"
    Write-Host "2. 手动配置 token:"
    Write-Host "   打开 https://www.kaggle.com/settings/api"
    Write-Host "   点击 Generate New Token，将 token 保存到:"
    Write-Host "   $AccessTokenPath"
    Write-Host "3. 或设置环境变量:"
    Write-Host '   $env:KAGGLE_API_TOKEN = "你的token"'
    Write-Host ""
    throw "缺少 Kaggle API 凭证，无法下载数据"
}

Write-Host "开始下载竞赛数据到: $DataDir"
Write-Host "数据较大 (约 6GB+)，请耐心等待..."

& $VenvPython -m kaggle competitions download -c drw-crypto-market-prediction -p $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle 下载失败"
}

$ZipPath = Join-Path $DataDir "drw-crypto-market-prediction.zip"
if (Test-Path $ZipPath) {
    Write-Host "解压压缩包..."
    Expand-Archive -Path $ZipPath -DestinationPath $DataDir -Force
}

$required = @("train.parquet", "test.parquet", "sample_submission.csv")
$missing = $required | Where-Object { -not (Test-Path (Join-Path $DataDir $_)) }
if ($missing.Count -gt 0) {
    throw "下载后仍缺少文件: $($missing -join ', ')"
}

Write-Host ""
Write-Host "数据下载完成:"
Get-ChildItem $DataDir | Format-Table Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}

Write-Host "运行数据校验..."
& $VenvPython (Join-Path $ProjectRoot "src\verify_data.py") --root $ProjectRoot
