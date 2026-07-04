# 异常波动样本加权建模实验报告

生成时间：2026-07-04T20:41:05

## 实验设置

- 验证划分：按时间顺序前 80% 训练、后 20% 验证。
- 验证分组：普通波动为 `abs(label)` 前 80%，中等波动为 80%-95%，异常波动为后 5%。
- 加权训练阈值仅由训练集 `abs(label)` 分位数计算。
- 目标模型：`连续权重 Weighted LightGBM`。
- 权重模式：`continuous`。
- 权重锚点：普通 `1.0`，中等 `1.5`，异常 `3.0`。
- 训练集权重阈值：80% 分位 `0.917035`，95% 分位 `1.974851`。
- Weighted LightGBM 最佳迭代轮数：`153`。

## 整体验证指标

| model_label | pearson | rmse | mae |
| --- | --- | --- | --- |
| Ridge | 0.091906 | 1.227520 | 0.868583 |
| LightGBM | 0.102359 | 1.039144 | 0.698482 |
| Weighted LightGBM | 0.072393 | 1.079566 | 0.741490 |
| 连续权重 Weighted LightGBM | 0.068236 | 1.056788 | 0.720159 |

## 异常波动组指标

| model_label | pearson | rmse | mae | squared_error_contribution | absolute_error_contribution |
| --- | --- | --- | --- | --- | --- |
| Ridge | 0.310821 | 3.284292 | 2.957680 | 35.79% | 17.03% |
| LightGBM | 0.269536 | 3.367959 | 3.149084 | 52.52% | 22.54% |
| Weighted LightGBM | 0.232541 | 3.356517 | 3.123331 | 48.33% | 21.06% |
| 连续权重 Weighted LightGBM | 0.234908 | 3.355173 | 3.130635 | 50.40% | 21.74% |

## LightGBM 与 Weighted LightGBM 误差贡献

| model_label | group | sample_ratio | squared_error_contribution | absolute_error_contribution | rmse | mae |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 普通波动 | 80.00% | 18.88% | 47.38% | 0.504807 | 0.413716 |
| LightGBM | 中等波动 | 15.00% | 28.60% | 30.07% | 1.434747 | 1.400314 |
| LightGBM | 异常波动 | 5.00% | 52.52% | 22.54% | 3.367959 | 3.149084 |
| 连续权重 Weighted LightGBM | 普通波动 | 80.00% | 21.26% | 49.12% | 0.544764 | 0.442206 |
| 连续权重 Weighted LightGBM | 中等波动 | 15.00% | 28.34% | 29.14% | 1.452595 | 1.399026 |
| 连续权重 Weighted LightGBM | 异常波动 | 5.00% | 50.40% | 21.74% | 3.355173 | 3.130635 |

## 结果分析

- 相比普通 LightGBM，连续权重 Weighted LightGBM 的整体 Pearson 变化为 `-0.034122`。
- 异常波动组 RMSE 变化为 `-0.012785`，MAE 变化为 `-0.018449`。
- Weighted LightGBM 在异常波动组的部分误差指标出现下降，说明样本加权对异常样本有一定改善作用。
- 从误差贡献看，异常波动样本占验证集约 5%，但在树模型中贡献了明显更高比例的平方误差，说明异常行情是主要误差来源之一。

## 图表文件

- `E:\DRW\outputs\figures\extreme_volatility\label_abs_distribution.png`
- `E:\DRW\outputs\figures\extreme_volatility\group_rmse_compare.png`
- `E:\DRW\outputs\figures\extreme_volatility\group_mae_compare.png`
- `E:\DRW\outputs\figures\extreme_volatility\group_pearson_compare.png`
- `E:\DRW\outputs\figures\extreme_volatility\error_contribution.png`
- `E:\DRW\outputs\figures\extreme_volatility\weighted_lgbm_delta.png`
