# CatBoost / XGBoost / LightGBM 整体与异常组对比

生成时间：2026-07-04T20:50:17

## 实验设置

- 数据划分：按时间顺序前 80% 训练、后 20% 验证。
- 特征列：复用 `models/official_lgbm_features.json`。
- LightGBM：复用 `outputs/experiments/official_lgbm_valid_predictions.csv`。
- 异常组定义：验证集 `abs(label)` 大于 95% 分位数。
- CatBoost 最佳迭代：`63`。
- XGBoost 最佳迭代：`39`。

## 指标对比

| model_label | group | sample_ratio | pearson | rmse | mae | squared_error_contribution | absolute_error_contribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 整体 | 100.00% | 0.102359 | 1.039144 | 0.698482 | 100.00% | 100.00% |
| LightGBM | 异常波动 | 5.00% | 0.269536 | 3.367959 | 3.149084 | 52.52% | 22.54% |
| CatBoost baseline | 整体 | 100.00% | 0.079399 | 1.039405 | 0.698053 | 100.00% | 100.00% |
| CatBoost baseline | 异常波动 | 5.00% | 0.235338 | 3.379480 | 3.156194 | 52.86% | 22.61% |
| XGBoost baseline | 整体 | 100.00% | 0.099598 | 1.037860 | 0.697046 | 100.00% | 100.00% |
| XGBoost baseline | 异常波动 | 5.00% | 0.280266 | 3.374244 | 3.152818 | 52.85% | 22.62% |

## 简要结论

- 整体 Pearson 最高模型：`LightGBM`。
- 异常组 RMSE 最低模型：`LightGBM`。
- 异常组 Pearson 最高模型：`XGBoost baseline`。
