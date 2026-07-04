# DRW Crypto Market Prediction

本仓库用于课程大作业验收，主题来自 Kaggle 竞赛 **DRW - Crypto Market Prediction**。项目完成了两部分工作：

1. 官方短期价格变化回归预测任务：读取 DRW 原始数据，完成特征处理、模型训练、验证和 submission 生成。
2. 自命题扩展任务：围绕 `abs(label)` 较大的异常波动样本，比较不同模型在普通、中等、异常波动区间的误差结构，并测试样本加权训练的影响。

## 给老师和助教的验收入口

推荐解释器：

```powershell
E:\miniconda\envs\drw\python.exe
```

基础检查：

```powershell
cd E:\DRW
& E:\miniconda\envs\drw\python.exe src\check_env.py
& E:\miniconda\envs\drw\python.exe src\verify_data.py --root .
& E:\miniconda\envs\drw\python.exe -m compileall src
```

一键重新生成主要 submission：

```powershell
.\scripts\generate_all_submissions.ps1 -Python E:\miniconda\envs\drw\python.exe -Root .
```

该脚本会依次生成官方 Ridge、官方 LightGBM、树模型融合、时序扩展 LightGBM、新方案迁移版三类 submission。若需要强制重建新方案缓存，加 `-RebuildNewSolutionCache`；若只想快速生成新方案 submission，可加 `-SkipNewSolutionCv` 跳过迁移版 CV。

## 数据要求

本地原始数据位于：

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

当前本地数据概况：

| 文件 | 行数 | 列数 | 说明 |
|---|---:|---:|---|
| `train.parquet` | 525,886 | 787 | 含 `label` |
| `test.parquet` | 538,150 | 786 | 不含 `label` |
| `sample_submission.csv` | 538,150 | 2 | 列为 `ID`、`prediction` |

`train.parquet` 和 `test.parquet` 均超过 3GB，已通过 `.gitignore` 排除。若助教在新环境复现，需要先从 Kaggle 下载并放入上述路径。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `src/data_preprocessing` | 数据读取、基础市场特征、新方案 450 维特征缓存构建 |
| `src/prediction_task` | 官方任务模型训练、submission 生成、异常波动实验、新方案迁移版训练 |
| `src/eda` | 数据探索和特征相关性分析 |
| `src/feature_effectiveness` | 特征组消融、稳定特征和 medoid 特征实验 |
| `src/visualization` | 实验图表生成 |
| `scripts` | 环境、数据下载和一键生成 submission 脚本 |
| `models` | 已训练模型和特征列表 |
| `outputs/experiments` | 指标表、参数、实验摘要 |
| `outputs/figures` | 报告或答辩可用图表 |
| `outputs/submissions` | Kaggle submission 文件 |
| `data/external` | 新方案迁移版所需特征规格、时间过滤表、参考 hash |

## 官方预测任务结果

官方任务使用时间顺序切分，前 80% 训练，后 20% 验证，主要指标为 Pearson 相关系数。

| 方法 | 特征数 | 主要脚本 | Holdout Pearson | RMSE | MAE | 主要产物 |
|---|---:|---|---:|---:|---:|---|
| Ridge baseline | 792 | `src/prediction_task/train_baseline.py` | 0.09191 | 1.22752 | 0.86858 | `models/official_ridge.pkl` |
| 调参 LightGBM | 792 | `src/prediction_task/train_lgbm.py` | 0.10236 | 1.03914 | 0.69848 | `models/official_lgbm.txt`、`outputs/submissions/submission.csv` |
| CatBoost/XGBoost/LightGBM 融合 | 792 | `src/prediction_task/run_overnight_optimization.py --steps 3` | 0.10883 | 1.03795 | 0.69694 | `outputs/submissions/submission_overnight_step3_tree_blend.csv` |
| 时序扩展 LightGBM | 809 | `src/prediction_task/run_overnight_optimization.py --steps 4` | 0.09628 | 1.04107 | 0.69990 | `outputs/submissions/submission_overnight_step4_temporal.csv` |

补充说明：当前 `outputs/experiments/overnight/optimization_summary.csv` 是最近一次单独运行 step4 后写出的摘要，所以只列 baseline 和 step4；step3 的融合权重和指标保存在 `outputs/experiments/overnight/step3_blend_weight_search.csv`，当前最优权重为 LightGBM 0.2、CatBoost 0.6、XGBoost 0.2。

## 自命题扩展任务

扩展任务关注异常波动样本的误差结构。验证集按 `abs(label)` 分为三组：

| 组别 | 定义 | 样本数 | 占比 |
|---|---|---:|---:|
| 普通波动 | `abs(label) <= 1.04688866138` | 84,141 | 80.00% |
| 中等波动 | `1.04688866138 < abs(label) <= 2.08640890121` | 15,777 | 15.00% |
| 异常波动 | `abs(label) > 2.08640890121` | 5,259 | 5.00% |

关键结论来自 `outputs/experiments/extreme_volatility_continuous_weighted_lgbm_model_metrics.csv`：

| 方法 | 整体 Pearson | 异常组 Pearson | 异常组 RMSE | 异常组平方误差贡献 |
|---|---:|---:|---:|---:|
| Ridge | 0.09191 | 0.31082 | 3.28429 | 35.79% |
| LightGBM | 0.10236 | 0.26954 | 3.36796 | 52.52% |
| Weighted LightGBM | 0.07239 | 0.23254 | 3.35652 | 48.33% |
| 连续权重 Weighted LightGBM | 0.06824 | 0.23491 | 3.35517 | 50.40% |

结论是：异常波动样本只占验证集 5%，但对平方误差贡献很高；样本加权可以略微降低异常组 RMSE/MAE，但会牺牲整体 Pearson，因此不作为最终官方任务提交模型，只作为自命题分析结果。

相关代码和产物：

- `src/prediction_task/analyze_extreme_volatility.py`
- `src/prediction_task/compare_tree_baselines_extreme.py`
- `src/prediction_task/compare_tree_blend_extreme.py`
- `src/prediction_task/train_two_stage_expert.py`
- `outputs/experiments/extreme_volatility_*`
- `outputs/figures/extreme_volatility/*`

## 新方案迁移版

迁移版使用 `data/external/new_solution_feature_spec.json` 和 `data/external/new_solution_time_filter.csv`，构建 450 维特征，时间过滤后训练样本为 71,282 行，测试样本为 538,150 行。

复现命令：

```powershell
& E:\miniconda\envs\drw\python.exe src\data_preprocessing\build_new_solution_dataset.py --raw-data-dir data\raw --asset-dir data\external --cache-dir data\processed\new_solution
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_new_solution.py --models linear,ridge,lightgbm --cache-dir data\processed\new_solution --output-dir outputs\experiments\new_solution --model-dir models\new_solution --submission-dir outputs\submissions --make-submissions
& E:\miniconda\envs\drw\python.exe src\prediction_task\verify_new_solution_migration.py --reference-manifest data\external\new_solution_reference_manifest.json --cache-dir data\processed\new_solution --output-dir outputs\experiments\new_solution --submission-dir outputs\submissions
```

迁移验收标准是缓存、OOF、submission 的 SHA256 和参考清单一致。当前 `verify_new_solution_migration.py` 会直接输出“迁移验收通过”或错误列表。

| 方法 | CV Pearson 均值 | submission | SHA256 |
|---|---:|---|---|
| LinearRegression | 不做固定时间 CV | `outputs/submissions/submission_new_solution_linear.csv` | `4074ea46dde6ad39e01b155a1fed883297d5767576b0d3698abf55a42fe3def4` |
| Ridge(alpha=100) | 0.79628 | `outputs/submissions/submission_new_solution_ridge.csv` | `a09938e9be79736345e4eb687b8757487b0431512a934127286453559f3ecad1` |
| LightGBM | 0.46950 | `outputs/submissions/submission_new_solution_lightgbm.csv` | `13081ed6056080cc1ddceaeca1defc56648ba10b20fceed9b7b6c1ee6f3fde71` |

## 轻量烟测

若只检查脚本链路，不想完整读取 3GB 级数据，可运行：

```powershell
$tmp = Join-Path $env:TEMP "drw_smoke"
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_baseline.py --root . --sample-rows 5000 --output-dir $tmp --model-dir $tmp
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_lgbm.py --root . --sample-rows 5000 --num-boost-round 20 --early-stopping-rounds 5 --output-dir $tmp --model-dir $tmp
& E:\miniconda\envs\drw\python.exe src\prediction_task\run_overnight_optimization.py --root . --steps 3,4 --sample-rows 5000 --smoke-test --output-dir (Join-Path $tmp "overnight")
& E:\miniconda\envs\drw\python.exe src\data_preprocessing\build_new_solution_dataset.py --sample-train-rows 20000 --sample-test-rows 1000 --cache-dir (Join-Path $tmp "new_solution_cache") --force
& E:\miniconda\envs\drw\python.exe src\prediction_task\train_new_solution.py --models linear,ridge,lightgbm --smoke-test --make-submissions --cache-dir (Join-Path $tmp "new_solution_cache") --output-dir (Join-Path $tmp "new_solution_out") --model-dir (Join-Path $tmp "new_solution_models") --submission-dir (Join-Path $tmp "new_solution_subs")
```

## 主要产物索引

- 官方最终提交：`outputs/submissions/submission.csv`
- 官方 Ridge 提交：`outputs/submissions/submission_official_ridge_baseline.csv`
- 官方 LightGBM 提交：`outputs/submissions/submission_official_lightgbm_tuned.csv`
- 树模型融合提交：`outputs/submissions/submission_overnight_step3_tree_blend.csv`
- 时序扩展提交：`outputs/submissions/submission_overnight_step4_temporal.csv`
- 新方案迁移版提交：`outputs/submissions/submission_new_solution_*.csv`
- 官方模型对比：`outputs/experiments/official_model_compare.csv`
- 异常波动指标：`outputs/experiments/extreme_volatility_continuous_weighted_lgbm_model_metrics.csv`
- 特征组消融：`outputs/experiments/feature_group_cv_summary.csv`
- 特征筛选流水线：`outputs/experiments/feature_pipeline_cv_summary.csv`
- 新方案迁移版摘要：`outputs/experiments/new_solution/run_summary.json`

## 环境依赖

依赖见 `requirements.txt`。主要库包括：

- `pandas`、`numpy`、`pyarrow`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `catboost`
- `matplotlib`、`seaborn`

本仓库当前推荐使用 `E:\miniconda\envs\drw\python.exe`，普通 `python` 可能不包含完整依赖。

## 注意事项

- 本地 holdout、时间 CV 和 Kaggle LB 不是完全一致的评价口径，报告或答辩中不应把本地 holdout 直接等同于最终榜单表现。
- 自命题异常波动实验的重点是误差结构分析和加权训练效果，不是替代官方最终提交。
- `kaggle.json` 或任何账号 token 不应放入项目目录。
- 本仓库现在只保留 `README.md` 作为说明文档，其它 Markdown 报告已删除；实验细节以代码、CSV/JSON 指标表和图表产物为准。
