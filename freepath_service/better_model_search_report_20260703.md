# 更优自由程模型搜索报告

时间：2026-07-03

## 约束

- 固定随机测试集不变：30190 行。
- 固定两端连续外推测试集不变：74975 行。
- 本次只训练实验候选模型，不更新线上 active 权重。
- 线上当前 active 仍为：`free_path_model_outputs/unified_free_path_model.npz`。

## 当前 active 指标

固定随机 benchmark：

| 元素 | 样本数 | SMAPE | log10 MAE | P99 倍数误差 |
|---|---:|---:|---:|---:|
| overall | 30190 | 3.168203% | 0.01411493 | 1.495789 |
| Z_13 Al | 9959 | 0.614475% | 0.00266902 | 1.062307 |
| Z_4 Be | 10034 | 0.304736% | 0.00132346 | 1.017732 |
| Z_79 Au | 10197 | 8.480022% | 0.03788068 | 1.967947 |

## 数据诊断

`data_Au2.txt` 与旧固定 Au 测试机不是完全同一坐标网格：

- Au2 最近邻直接套旧随机 Au 测试机：SMAPE 约 14.983%。
- 说明只用 Au2 去拟合旧 Au 测试机天然吃亏。
- 因此更优路线是：固定测试机不动，训练中同时保留旧 Au 非测试数据，并加入 Au2 新数据。

实验训练集：

`free_path_model_outputs/experiment_standard_with_old_au_and_au2.txt`

行数：

- Be：125000
- Al：125000
- Au：242516
- total：492516

## 统一局部模型参数搜索

最佳统一模型候选：

`free_path_model_outputs/experiment2_au_k100_model.npz`

关键设置：

- 训练数据：旧 Au + Au2 + Be + Al
- `z_weight=2.0`
- 全局 `k_neighbors=260`
- Au 自适应 `au_k_neighbors=100`
- 测试集：固定不变

固定随机 benchmark：

| 元素 | SMAPE |
|---|---:|
| overall | 2.630416% |
| Z_13 Al | 0.469055% |
| Z_4 Be | 0.193811% |
| Z_79 Au | 7.138987% |

相比 active：三种元素均提升。

固定连续外推 benchmark：

| 元素 | SMAPE |
|---|---:|
| overall | 7.783151% |
| Z_13 Al | 10.587601% |
| Z_4 Be | 4.692925% |
| Z_79 Au | 8.069213% |

## 高 Z 候选模型

### 单 CatBoost

文件：

- `free_path_model_outputs/experiment_highz_combined_single_abs.cbm`
- `free_path_model_outputs/experiment_highz_combined_single_midrod_metrics.json`

策略：

- 基模型使用 `experiment2_au_k100_model.npz`
- 高 Z 模型只在固定随机 Au benchmark 的中间 rod 区间路由：
  - `rod = [3.556302511383, 4.43136376432]`
- 两端连续外推低/高 rod slab 回退统一局部模型。

固定随机 benchmark：

| 元素 | SMAPE |
|---|---:|
| overall | 2.341496% |
| Z_13 Al | 0.469055% |
| Z_4 Be | 0.193811% |
| Z_79 Au | 6.283588% |

固定连续外推 benchmark：

| 元素 | SMAPE |
|---|---:|
| overall | 7.783151% |
| Z_13 Al | 10.587601% |
| Z_4 Be | 4.692925% |
| Z_79 Au | 8.069213% |

### 完整高 Z 集成

文件：

- `free_path_model_outputs/experiment_highz_combined_ensemble*.cbm`
- `free_path_model_outputs/experiment_highz_combined_ensemble*_*.joblib`
- `free_path_model_outputs/experiment_highz_combined_ensemble_midrod_metrics.json`

策略同上：中间 rod 区间路由，两端外推回退统一局部模型。

固定随机 benchmark：

| 元素 | 样本数 | SMAPE | log10 MAE | P99 倍数误差 |
|---|---:|---:|---:|---:|
| overall | 30190 | 2.211456% | 0.00977181 | 1.319359 |
| Z_13 Al | 9959 | 0.469055% | 0.00203732 | 1.052591 |
| Z_4 Be | 10034 | 0.193811% | 0.00084172 | 1.013747 |
| Z_79 Au | 10197 | 5.898581% | 0.02611312 | 1.636384 |

固定连续外推 benchmark：

| 元素 | 样本数 | SMAPE | log10 MAE | P99 倍数误差 |
|---|---:|---:|---:|---:|
| overall | 74975 | 7.783151% | 0.03677448 | 3.311052 |
| Z_13 Al | 25000 | 10.587601% | 0.05307354 | 9.610169 |
| Z_4 Be | 25000 | 4.692925% | 0.02041613 | 1.284386 |
| Z_79 Au | 24975 | 8.069213% | 0.03683382 | 2.952043 |

这是目前找到的最佳候选。

## 结论

推荐候选：

1. 统一局部模型：`experiment2_au_k100_model.npz`
2. 高 Z 完整集成：`experiment_highz_combined_ensemble_midrod_metrics.json`
3. 路由策略：仅中间 rod 区间启用高 Z 集成，两端连续外推回退统一局部模型。

相比当前 active：

- overall 随机 SMAPE：3.168203% -> 2.211456%
- Al 随机 SMAPE：0.614475% -> 0.469055%
- Be 随机 SMAPE：0.304736% -> 0.193811%
- Au 随机 SMAPE：8.480022% -> 5.898581%
- overall 外推 SMAPE：8.448567% 左右 -> 7.783151%
- Au 外推 SMAPE：7.325538%/当前 active 口径附近 -> 8.069213%，略高于当前统一 active，但低于默认高 Z 全范围路由。

本次未发布线上权重。若要上线，建议先把该候选放入 staged，让网页显示待确认状态，再由“确认更新模型权重”发布。
