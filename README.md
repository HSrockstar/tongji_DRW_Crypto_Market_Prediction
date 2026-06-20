# DRW Crypto Market Prediction

## 项目信息

- 项目名称：DRW Crypto Market Prediction
- 项目路径：`E:\DRW`
- 当前阶段：环境配置与数据校验

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

推荐使用已有 conda 环境：

```powershell
conda activate drw
pip install -r requirements.txt
```

如果 PowerShell 中 `conda activate drw` 因本地 PATH 编码问题失败，可以直接使用环境内 Python：

```powershell
E:\miniconda\envs\drw\python.exe E:\DRW\src\check_env.py
```

## 当前已完成事项

- 目录创建
- conda 环境检查
- Kaggle 数据下载或确认
- 数据文件解压或确认
- 环境校验脚本：`src/check_env.py`
- 数据校验脚本：`src/verify_data.py`
- 校验报告：`outputs/data_check_report.json`
- GitHub 上传前的 `.gitignore` 与 `requirements.txt`

## 校验方法

```powershell
conda activate drw
python E:\DRW\src\check_env.py
python E:\DRW\src\verify_data.py --root E:\DRW
python E:\DRW\src\verify_data.py --root E:\DRW --hash
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

1. EDA
2. 时间顺序验证集划分
3. baseline 模型
4. 生成 `submission.csv`
5. 上传 Kaggle 并保存成绩截图
