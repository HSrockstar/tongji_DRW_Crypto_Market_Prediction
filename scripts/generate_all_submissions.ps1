param(
    [string]$Python = "E:\miniconda\envs\drw\python.exe",
    [string]$Root = ".",
    [switch]$RebuildSecondPlaceCache,
    [switch]$SkipSecondPlaceCv
)

$ErrorActionPreference = "Stop"

try {
    chcp 65001 | Out-Null
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$rootPath = Resolve-Path $Root
Set-Location $rootPath

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "==== $Name ===="
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name 失败，退出码 $LASTEXITCODE"
    }
}

if (-not (Test-Path $Python)) {
    throw "找不到 Python 解释器：$Python"
}

$requiredFiles = @(
    "data\raw\train.parquet",
    "data\raw\test.parquet",
    "data\raw\sample_submission.csv",
    "models\official_ridge.pkl",
    "models\official_ridge_features.json",
    "models\official_lgbm.txt",
    "models\official_lgbm_features.json",
    "outputs\experiments\lgbm_best_params.json",
    "data\external\second_place_feature_spec.json",
    "data\external\second_place_time_filter.csv"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        throw "缺少必要文件：$file"
    }
}

New-Item -ItemType Directory -Force -Path "outputs\submissions" | Out-Null

Invoke-Step "官方 Ridge baseline submission" @(
    "src\prediction_task\make_submission.py",
    "--root", ".",
    "--model", "models\official_ridge.pkl",
    "--model-type", "ridge",
    "--output", "outputs\submissions\submission_official_ridge_baseline.csv"
)

Invoke-Step "官方调参 LightGBM submission" @(
    "src\prediction_task\make_submission.py",
    "--root", ".",
    "--model", "models\official_lgbm.txt",
    "--model-type", "lightgbm",
    "--output", "outputs\submissions\submission_official_lightgbm_tuned.csv"
)

Invoke-Step "Step3 CatBoost + XGBoost + LightGBM 树模型融合 submission" @(
    "src\prediction_task\run_overnight_optimization.py",
    "--root", ".",
    "--steps", "3"
)

Invoke-Step "Step4 时序扩展 LightGBM submission" @(
    "src\prediction_task\run_overnight_optimization.py",
    "--root", ".",
    "--steps", "4"
)

$secondPlaceCacheArgs = @(
    "src\data_preprocessing\build_second_place_dataset.py",
    "--raw-data-dir", "data\raw",
    "--asset-dir", "data\external",
    "--cache-dir", "data\processed\second_place"
)
if ($RebuildSecondPlaceCache) {
    $secondPlaceCacheArgs += "--force"
}
Invoke-Step "第二名迁移版 450 特征缓存" $secondPlaceCacheArgs

$secondPlaceTrainArgs = @(
    "src\prediction_task\train_second_place.py",
    "--models", "linear,ridge,lightgbm",
    "--cache-dir", "data\processed\second_place",
    "--output-dir", "outputs\experiments\second_place",
    "--model-dir", "models\second_place",
    "--submission-dir", "outputs\submissions",
    "--make-submissions"
)
if ($SkipSecondPlaceCv) {
    $secondPlaceTrainArgs += "--no-cv"
}
Invoke-Step "第二名迁移版 linear/ridge/lightgbm submissions" $secondPlaceTrainArgs

Write-Host ""
Write-Host "全部 submission 已生成："
$expectedSubmissions = @(
    "outputs\submissions\submission_official_ridge_baseline.csv",
    "outputs\submissions\submission_official_lightgbm_tuned.csv",
    "outputs\submissions\submission_overnight_step3_tree_blend.csv",
    "outputs\submissions\submission_overnight_step4_temporal.csv",
    "outputs\submissions\submission_second_place_linear.csv",
    "outputs\submissions\submission_second_place_ridge.csv",
    "outputs\submissions\submission_second_place_lightgbm.csv"
)

foreach ($submission in $expectedSubmissions) {
    if (Test-Path $submission) {
        $item = Get-Item $submission
        Write-Host ("- {0} ({1:N0} bytes)" -f $submission, $item.Length)
    } else {
        Write-Host "- 未生成：$submission"
    }
}
