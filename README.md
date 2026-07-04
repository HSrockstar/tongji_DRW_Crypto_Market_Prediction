# DRW Crypto Market Prediction

## 项目信息

- 项目名称：DRW Crypto Market Prediction
- 当前阶段：主线方法已精简为 Ridge baseline、调参 LightGBM、CatBoost + XGBoost 树模型集成、时序扩展 LightGBM；另保留第 2 名方案迁移版作为独立可复现实验。
- 验证口径：本地时间 Holdout / 时间 CV / Kaggle LB 均保留，但报告中需明确它们不完全一致。
- 推荐解释器：`E:\miniconda\envs\drw\python.exe`。当前默认 `python` 环境依赖不完整。

## 目录结构

- `data/raw`：Kaggle 原始数据文件。
- `data/processed`：后续处理后的数据文件。
- `src`：项目脚本。
- `models`：保留主线模型文件。
- `outputs`：校验报告、实验结果、图表和提交文件。
- `data/external`：第 2 名方案迁移版所需的时间过滤与特征规格资产。

## 数据文件

解压后应包含：

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

已确认数据概况：

- `train.parquet`：525886 行，787 列，含 `label`。
- `test.parquet`：538150 行，786 列。
- `sample_submission.csv`：列名为 `ID`、`prediction`。
- 公开市场字段：`bid_qty`、`ask_qty`、`buy_qty`、`sell_qty`、`volume`。

`train.parquet` 和 `test.parquet` 均超过 3GB，已通过 `.gitignore` 排除。

## 当前保留方法

| 方法 | 使用特征 | 脚本 | 当前表现 |
|------|----------|------|----------|
| Ridge baseline | 792 维完整特征 | `src/prediction_task/train_baseline.py` | Holdout Pearson 0.0914 |
| 调参 LightGBM 主 baseline | 792 维完整特征 | `src/prediction_task/tune_lgbm.py`、`train_lgbm.py` | Holdout Pearson 0.1024 |
| CatBoost + XGBoost 树模型集成 | 792 维完整特征 | `src/prediction_task/run_overnight_optimization.py --steps 3` | Holdout Pearson 0.1134，Kaggle 泛化差 |
| 时序扩展 LightGBM | 约 809 维时序扩展特征 | `src/prediction_task/run_overnight_optimization.py --steps 4` | Holdout Pearson 0.0963，Private LB 较稳 |

特征组消融、medoid 与稳定特征筛选保留为特征贡献分析，不再作为最终建模方法。

## 第 2 名方案迁移版

第 2 名方案已经迁移到主项目，包含三条独立方法：原始 `LinearRegression()`、`Ridge(alpha=100.0)`、`LightGBM`。迁移实现不保留 Kaggle notebook、metadata、下载脚本或复制来的原始代码，只保留必要的数据资产和重写后的工程化实现。

| 方法 | 数据/特征 | 脚本 | 验收结果 |
|------|----------|------|----------|
| 原始线性模型 | 时间过滤后 71,282 行训练样本，450 维特征 | `src/prediction_task/train_second_place.py --models linear` | 提交 SHA 与迁移前参考完全一致 |
| Ridge 迁移版 | 同上 | `src/prediction_task/train_second_place.py --models ridge` | 提交 SHA 与迁移前参考完全一致 |
| LightGBM 迁移版 | 同上，最终 `n_estimators=1627` | `src/prediction_task/train_second_place.py --models lightgbm` | 提交 SHA 与迁移前参考完全一致 |

## 运行命令

```powershell
cd E:\DRW

# Ridge baseline
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_baseline.py --root .

# 调参 LightGBM
& E:\miniconda\envs\drw\python.exe src\prediction_task\tune_lgbm.py --root .

# 树模型集成 + 时序扩展 LightGBM
& E:\miniconda\envs\drw\python.exe src\prediction_task\run_overnight_optimization.py --root . --steps 3,4

# 生成 LightGBM 提交
& E:\miniconda\envs\drw\python.exe src\prediction_task\make_submission.py --root . --model models\official_lgbm.txt

# 第 2 名方案迁移版：构建缓存、训练三条方法并验收
& E:\miniconda\envs\drw\python.exe src\data_preprocessing\build_second_place_dataset.py --raw-data-dir data\raw --asset-dir data\external --cache-dir data\processed\second_place
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_second_place.py --models linear,ridge,lightgbm --cache-dir data\processed\second_place --output-dir outputs\experiments\second_place --model-dir models\second_place --submission-dir outputs\submissions --make-submissions
& E:\miniconda\envs\drw\python.exe src\prediction_task\verify_second_place_migration.py --reference-manifest data\external\second_place_reference_manifest.json --cache-dir data\processed\second_place --output-dir outputs\experiments\second_place --submission-dir outputs\submissions
```

轻量烟测：

```powershell
$tmp = Join-Path $env:TEMP "drw_smoke"
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_baseline.py --root . --sample-rows 5000 --output-dir $tmp --model-dir $tmp
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_lgbm.py --root . --sample-rows 5000 --num-boost-round 20 --early-stopping-rounds 5 --output-dir $tmp --model-dir $tmp
& E:\miniconda\envs\drw\python.exe src\prediction_task\run_overnight_optimization.py --root . --steps 3,4 --sample-rows 5000 --smoke-test --output-dir (Join-Path $tmp "overnight")
& E:\miniconda\envs\drw\python.exe src\data_preprocessing\build_second_place_dataset.py --sample-train-rows 20000 --sample-test-rows 1000 --cache-dir (Join-Path $tmp "second_place_cache") --force
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_second_place.py --models linear,ridge,lightgbm --smoke-test --make-submissions --cache-dir (Join-Path $tmp "second_place_cache") --output-dir (Join-Path $tmp "second_place_out") --model-dir (Join-Path $tmp "second_place_models") --submission-dir (Join-Path $tmp "second_place_subs")
```

## 校验方法

```powershell
& E:\miniconda\envs\drw\python.exe src\check_env.py
& E:\miniconda\envs\drw\python.exe src\verify_data.py --root .
& E:\miniconda\envs\drw\python.exe -m compileall src
```

## 关键产物

- `models/official_ridge.pkl`
- `models/official_lgbm.txt`
- `outputs/submissions/submission.csv`
- `outputs/experiments/official_model_compare.csv`
- `outputs/experiments/overnight/optimization_summary.csv`
- `outputs/experiments/feature_group_cv_summary.csv`
- `outputs/experiments/feature_pipeline_cv_summary.csv`
- `data/external/second_place_feature_spec.json`
- `data/external/second_place_reference_manifest.json`

## 注意事项

- 本地 Holdout 与 Kaggle LB 存在明显偏差，不能只根据 Holdout 选最终方案。
- 第 2 名方案迁移版不保留 Kaggle 原始代码；验收以主项目生成产物与迁移前参考 SHA/指标完全一致为准。
- 不要把 `kaggle.json` 或任何账号 token 放进项目目录或提交到 GitHub。
