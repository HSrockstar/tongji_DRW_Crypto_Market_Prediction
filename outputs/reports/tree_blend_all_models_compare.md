# Tree Blend 融合模型对比报告

生成时间：2026-07-04T20:58:55

## 实验设置

- 融合模型：`Tree Blend = LightGBM + CatBoost + XGBoost`。
- 权重搜索目标：最小化异常波动组 RMSE。
- 权重约束：三个权重非负，且权重和为 1。
- 搜索步长：`0.05`。
- 异常组定义：验证集 `abs(label)` 大于 95% 分位数。

## 最优融合权重

| base_model | weight |
| --- | ---: |
| LightGBM | 1.00 |
| CatBoost baseline | 0.00 |
| XGBoost baseline | 0.00 |

权重搜索最优行：

| overall_pearson | overall_rmse | overall_mae | extreme_pearson | extreme_rmse | extreme_mae |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.102359 | 1.039144 | 0.698482 | 0.269536 | 3.367959 | 3.149084 |

## 全模型指标对比

| model_label | group | sample_ratio | pearson | rmse | mae | squared_error_contribution | absolute_error_contribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | 整体 | 100.00% | 0.091906 | 1.227520 | 0.868583 | 100.00% | 100.00% |
| Ridge | 异常波动 | 5.00% | 0.310821 | 3.284292 | 2.957680 | 35.79% | 17.03% |
| LightGBM | 整体 | 100.00% | 0.102359 | 1.039144 | 0.698482 | 100.00% | 100.00% |
| LightGBM | 异常波动 | 5.00% | 0.269536 | 3.367959 | 3.149084 | 52.52% | 22.54% |
| Weighted LightGBM | 整体 | 100.00% | 0.072393 | 1.079566 | 0.741490 | 100.00% | 100.00% |
| Weighted LightGBM | 异常波动 | 5.00% | 0.232541 | 3.356517 | 3.123331 | 48.33% | 21.06% |
| 连续权重 Weighted LightGBM | 整体 | 100.00% | 0.068236 | 1.056788 | 0.720159 | 100.00% | 100.00% |
| 连续权重 Weighted LightGBM | 异常波动 | 5.00% | 0.234908 | 3.355173 | 3.130635 | 50.40% | 21.74% |
| CatBoost baseline | 整体 | 100.00% | 0.079399 | 1.039405 | 0.698053 | 100.00% | 100.00% |
| CatBoost baseline | 异常波动 | 5.00% | 0.235338 | 3.379480 | 3.156194 | 52.86% | 22.61% |
| XGBoost baseline | 整体 | 100.00% | 0.099598 | 1.037860 | 0.697046 | 100.00% | 100.00% |
| XGBoost baseline | 异常波动 | 5.00% | 0.280266 | 3.374244 | 3.152818 | 52.85% | 22.62% |
| Tree Blend | 整体 | 100.00% | 0.102359 | 1.039144 | 0.698482 | 100.00% | 100.00% |
| Tree Blend | 异常波动 | 5.00% | 0.269536 | 3.367959 | 3.149084 | 52.52% | 22.54% |

## 简要结论

- 整体 Pearson 最高模型：`LightGBM`。
- 整体 RMSE 最低模型：`XGBoost baseline`。
- 异常组 RMSE 最低模型：`Ridge`。
- 异常组 Pearson 最高模型：`Ridge`。
- 按异常组 RMSE 优化时，最优 Tree Blend 退化为纯 LightGBM，说明当前非负加权融合未进一步降低异常组 RMSE。
- 本次融合权重直接在当前验证集上搜索，用于课程实验分析，不作为严格无偏泛化估计。

## 图表文件

- `E:\DRW\outputs\figures\extreme_volatility\tree_blend_overall_metrics.png`
- `E:\DRW\outputs\figures\extreme_volatility\tree_blend_extreme_metrics.png`
- `E:\DRW\outputs\figures\extreme_volatility\tree_blend_weights.png`
