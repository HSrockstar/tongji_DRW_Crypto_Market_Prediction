# DRW Crypto Market Prediction

## 项目信息

- 项目名称：DRW Crypto Market Prediction
- 项目路径：以仓库根目录为准（本地示例：`D:\数学建模期末大作业\tongji_DRW_Crypto_Market_Prediction`）
- 当前阶段：官方预测 baseline 已完成；5 折时间 CV 与特征组消融已有初版结果；报告/PPT/视频待撰写
- 最近同步：`origin/main` @ `b96cbb1`（7.3 增加交叉验证）

## 目录结构

- `data/raw`：Kaggle 原始数据文件
- `data/processed`：后续处理后的数据文件
- `notebooks`：探索性分析 Notebook
- `src`：项目脚本
- `outputs`：校验报告、分析输出等
- `models`：后续模型文件
- `logs`：运行日志

## 数据文件说明

- `train.parquet` 是训练集，包含特征和目标列。
- `test.parquet` 是测试集，用来生成预测结果。
- `sample_submission.csv` 是 Kaggle 提交格式模板。

`train.parquet` 和 `test.parquet` 文件都超过 3GB，不适合直接提交到 GitHub，已通过 `.gitignore` 排除。当前仓库只计划提交代码、文档、依赖文件、校验脚本、校验报告和较小的 `sample_submission.csv`。

## 数据获取

在已经配置好 Kaggle API 的环境中执行：

```powershell
kaggle competitions download -c drw-crypto-market-prediction -p E:\DRW\data\raw
```

下载完成后，将压缩包解压到：

```text
E:\DRW\data\raw
```

解压后应包含：

```text
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

不要把 `kaggle.json` 或任何账号 token 放进项目目录或提交到 GitHub。

## 环境依赖

推荐在本机使用虚拟环境（Windows 示例）：

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

也可手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\check_env.py
```

## 当前已完成事项

### 环境与数据

- 目录结构与依赖：`requirements.txt`、`.gitignore`
- 本地虚拟环境：`.venv`（本机）
- Kaggle 竞赛已加入，原始数据已下载并校验：`data/raw/train.parquet`、`test.parquet`
- 环境/数据脚本：`src/check_env.py`、`src/verify_data.py`
- 本机辅助脚本（未提交）：`scripts/setup_env.ps1`、`scripts/download_data.ps1`
- 校验报告：`outputs/data_check_report.json`

### 任务一：官方预测

- 数据预处理与基础特征：`src/data_preprocessing/`
- 单次 holdout 验证：Ridge / LightGBM / Lasso（`outputs/experiments/official_*`）
- 提交文件：`outputs/submissions/submission.csv`
- 已保存模型：`models/official_ridge.pkl`、`official_lgbm.txt`、`official_lasso.pkl`

### 任务二：特征有效性（初版）

- 5 折 walk-forward 特征组消融（Ridge）：`outputs/experiments/feature_group_cv_*.csv`
- 对比图：`outputs/figures/feature_effectiveness/feature_group_compare.png`
- 待补充：LightGBM importance / SHAP、更完整的 G1~G6 实验说明

### 任务三：时间稳定性（初版）

- 5 折 expanding window CV：`src/prediction_task/run_time_cv_experiments.py`
- 结果：`outputs/experiments/cv_results.csv`、`cv_summary.csv`
- 趋势图：`outputs/figures/prediction_task/cv_pearson_by_fold.png`

### 待完成

- Kaggle 正式提交与成绩截图
- 报告 PDF、PPT、约 8 分钟汇报视频
- 特征解释与时间稳定性章节的文字分析

## 校验方法

```powershell
.\.venv\Scripts\Activate.ps1
python src\check_env.py
python src\verify_data.py --root .
python src\verify_data.py --root . --hash
```

## 已确认的数据概况

- `train.parquet`：525886 行，787 列
- `test.parquet`：538150 行，786 列
- `sample_submission.csv`：列名为 `ID`、`prediction`
- 常见市场字段：`bid_qty`、`ask_qty`、`buy_qty`、`sell_qty`、`volume`

## GitHub 提交范围

计划提交：

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/check_env.py`
- `src/verify_data.py`
- `outputs/data_check_report.json`
- `data/raw/sample_submission.csv`

不提交：

- `data/raw/train.parquet`
- `data/raw/test.parquet`
- `data/raw/drw-crypto-market-prediction.zip`
- `kaggle.json`
- 模型文件和本地日志

## 下一步建议

1. 用最新全量数据重跑 CV / 特征消融，确认结果可复现
2. 补充 LightGBM 特征重要性与 SHAP 分析（任务二）
3. 整理时间 CV 图表与文字结论（任务三）
4. 提交 Kaggle 并保存成绩截图
5. 撰写 PDF 报告、PPT 与汇报视频
