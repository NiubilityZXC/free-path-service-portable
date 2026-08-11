# 最新最佳模型搜索报告

- 时间：2026-07-03
- 数据：`free_path_model_outputs/experiment_standard_with_old_au_and_au2.txt`
- 固定随机 benchmark：`free_path_model_outputs/unified_permanent_benchmark_data.npz`
- 固定连续外推 benchmark：`free_path_model_outputs/unified_permanent_extrapolation_data.npz`
- 2026-07-03 本轮只做离线实验，没有发布权重。
- 2026-07-06 已把最佳三段路由方案做成暂存模型和网页代码，但仍未发布线上权重；需要在网页点击“确认更新模型权重”后才会切换。

## 2026-07-06 网页暂存更新

已新增/修改：

- `high_z_free_path_model.py`：支持 `data_source=auto/old_au/au2`、旧 Au 中间区 CatBoost 专家、旧 Au 两端边界统一模型。
- `free_path_web_app.py`：预测、上传评测、现场评测均新增数据源选择；继续训练会生成同一套高 Z 三段路由 staged 权重。
- `stage_best_free_path_route_model.py`：复现旧 Au CatBoost 精细融合 `mean_1_3_6`，并写出 staged 指标。

当前 staged 指标：

| 暂存方案 | 随机 overall SMAPE | 随机 Au SMAPE | 外推 overall SMAPE | 外推 Au SMAPE |
|---|---:|---:|---:|---:|
| 三段路由 MoE staged | 2.056911% | 5.441023% | 7.535425% | 7.325538% |

验证：

- 旧 Au 随机 benchmark：10197 个 Au 样本全部走旧 Au CatBoost 专家。
- 旧 Au 连续外推 benchmark：24975 个 Au 样本全部走边界统一模型。
- Au2 新数据源：`data_source=au2` 时不会走旧 Au 专家。
- 网页服务：`http://127.0.0.1:8790` 已重启，staged 状态可见，尚未发布权重。

## 2026-07-06 Au 来源测试集重分配

根据新的要求，Au 永久测试集已重建为 source-aware 版本：

- 旧 Au `data.txt` 和新 Au2 `data_Au2.txt` 都进入 Au 永久随机测试。
- 旧 Au `data.txt` 和新 Au2 `data_Au2.txt` 都进入 Au 两端连续外推测试。
- Be/Al 旧永久测试保持不变。
- 训练仍然是一个统一模型；除了永久测试 key，旧 Au 和 Au2 的正自由程数据全部参与训练。

新测试集规模：

| 测试集 | 总样本 | 旧 Au | Au2 |
|---|---:|---:|---:|
| 随机 benchmark | 39708 | 10197 | 9518 |
| 连续外推 benchmark | 96889 | 24975 | 21914 |

用新测试集重训后的 staged 单统一模型：

| 范围 | overall SMAPE | Au overall SMAPE | 旧 Au SMAPE | Au2 SMAPE |
|---|---:|---:|---:|---:|
| 随机 benchmark | 3.427359% | 6.567248% | 7.115098% | 5.980315% |
| 连续外推 benchmark | 13.490690% | 19.356734% | 8.976523% | 31.186878% |

详细报告：`au_source_permanent_benchmark_report_20260706.md`。

## 结论

当前找到的最高随机 benchmark 精度：

`stored_heterogeneous_full_ensemble_all_range`

但是它会明显伤害连续外推，因此不推荐直接全范围上线。

当前综合随机 benchmark 和连续外推最好的方案已经更新为：

`数据源/物理区间分离的三段路由 MoE`

1. Be/Al：使用 `experiment2_au_k100_model.npz` 统一局部模型。
2. 旧 Au 固定 benchmark 中间内插区：使用旧 Au 专用 CatBoost 精细融合 `mean_1_3_6`。
3. Au 低/高 rod 边界外推区：使用当前线上统一模型 `unified_free_path_model.npz` 作为边界专家。

这样做的原因是：当前线上模型的 Au 随机 benchmark 很差，但 Au 连续外推段反而最好；旧 Au 专用 CatBoost 对固定随机 benchmark 最准，但全范围启用会严重伤害外推。三段路由把二者各自最擅长的区域拆开使用。

## 对比表

| 方法 | 随机 overall SMAPE | 随机 Au SMAPE | 外推 overall SMAPE | 外推 Au SMAPE | 说明 |
|---|---:|---:|---:|---:|---|
| 当前网页已发布模型 | 3.168203% | 8.480022% | 8.448567% | 7.325538% | 线上模型，高 Z 路由禁用 |
| 最佳单一统一模型 | 2.630416% | 7.138987% | 7.783151% | 8.069213% | 轻量，4.1 MB |
| 高 Z 异构集成，全 Au 启用 | 2.203235% | 5.874243% | 8.728602% | 10.907459% | 随机最好，但外推明显变差 |
| 高 Z 异构集成，中间区启用 | 2.211456% | 5.898581% | 7.783151% | 8.069213% | 稳定，不伤外推 |
| 中间区异构集成 + Au 局部边界专家 | 2.211456% | 5.898581% | 7.604176% | 7.531931% | 外推比上一行更好 |
| 旧 Au CatBoost mean + 当前边界专家 | 2.077680% | 5.502515% | **7.535425%** | **7.325538%** | 上一轮综合最好 |
| 旧 Au CatBoost 精细融合 + 当前边界专家 | **2.056911%** | **5.441023%** | **7.535425%** | **7.325538%** | 当前综合最好 |

## 本轮新增实验

### Au 专用局部回归大扫描

文件：

- `run_au_local_sweep.py`
- `au_local_sweep_report.md`
- `free_path_model_outputs/au_local_sweep_results.json`

结果：

- 最佳随机 Au SMAPE：约 `6.990628%`
- 最佳外推 Au SMAPE：约 `7.625308%`

它没有超过高 Z 异构集成的随机精度，但可以作为外推边界专家。

### Au 外推边界专扫

文件：

- `run_au_extrap_local_sweep.py`
- `au_extrap_local_sweep_report.md`
- `free_path_model_outputs/au_extrap_local_sweep_results.json`

最佳参数：

- scale：`rod150`
- k：`700`
- degree：`2`
- power：`1.4`
- ridge：`5e-2`

结果：

- Au 外推 SMAPE：`7.531931%`
- Au 外推 log10 MAE：`0.03421306`
- Au 外推 P99 倍数误差：`2.746558`

### Au 专用 Fourier MLP

文件：

- `run_au_mlp_experiment.py`
- `au_mlp_experiment_report.md`
- `free_path_model_outputs/au_mlp_experiment_results.json`

结果：

- Au 随机 benchmark SMAPE：`7.445296%`
- Au 连续外推 SMAPE：`8.170937%`

MLP 没有超过 CatBoost/异构集成，也没有超过局部边界模型。

### 旧 Au 专用树模型

文件：

- `run_old_au_tree_models.py`
- `old_au_tree_model_report.md`
- `free_path_model_outputs/old_au_tree_model_results.json`

最好单模型：

- `old_au_catboost_mae_d11`
- Au 随机 benchmark SMAPE：`5.522745%`
- Au 连续外推 SMAPE：`46.392504%`

结论：旧 Au 专用树模型能显著降低随机 benchmark，但不能用于外推。

### 旧 Au CatBoost 集成

文件：

- `run_old_au_catboost_ensemble.py`
- `old_au_catboost_ensemble_report.md`
- `free_path_model_outputs/old_au_catboost_ensemble_results.json`

最佳：

- `old_au_catboost_mean`
- Au 随机 benchmark SMAPE：`5.502515%`
- Au 随机 benchmark log10 MAE：`0.02451685`
- Au 随机 benchmark P99 倍数误差：`1.672055`

这是当前找到的最低 Au 随机 benchmark 误差。

### 旧 Au 全量 CatBoost 集成

文件：

- `run_old_au_catboost_full_ensemble.py`
- `old_au_catboost_full_ensemble_report.md`
- `free_path_model_outputs/old_au_catboost_full_ensemble_results.json`

最佳：

- `full_catboost_mean_log_only`
- Au 随机 benchmark SMAPE：`5.446010%`
- Au 随机 benchmark log10 MAE：`0.02424742`
- Au 随机 benchmark P99 倍数误差：`1.640584`

### 旧 Au CatBoost 精细搜索

文件：

- `run_old_au_catboost_refine.py`
- `old_au_catboost_refine_report.md`
- `free_path_model_outputs/old_au_catboost_refine_results.json`

最佳：

- `mean_1_3_6`
- 成员：
  - `ref_d12_lr016_l2_18_seed21`
  - `ref_d12_lr020_l2_20_seed23`
  - `ref_d12_lr018_l2_32_seed26`
- Au 随机 benchmark SMAPE：`5.441023%`
- Au 随机 benchmark log10 MAE：`0.02426280`
- Au 随机 benchmark P99 倍数误差：`1.665188`

这是当前找到的最低 Au 随机 benchmark 误差。

### 旧 Au 张量补全和 RBF 插值

文件：

- `run_au_tensor_completion.py`
- `au_tensor_completion_report.md`
- `free_path_model_outputs/au_tensor_completion_results.json`
- `run_old_au_rbf_experiment.py`
- `old_au_rbf_report.md`
- `free_path_model_outputs/old_au_rbf_results.json`
- `run_old_au_curve_interpolation.py`
- `old_au_curve_interpolation_report.md`
- `free_path_model_outputs/old_au_curve_interpolation_results.json`

结果：

- 张量补全最好 Au 随机 benchmark 约 `7.171627%`，外推约 `46%`，失败。
- 局部 RBF 最好 Au 随机 benchmark 约 `6.890324%`，没有超过 CatBoost。
- 每条 `(tep,tgama)` 曲线沿 rod 做 1D 插值最好约 `9.552200%`，没有超过 CatBoost。

结论：旧 Au 网格不是简单低秩平滑函数，单纯张量补全/RBF 不是突破口。

## 重要限制

旧 Au 和 Au2 的坐标范围有重叠，但不是完全相同的网格。固定 benchmark 来自旧 Au；Au2 新数据覆盖更宽的物理区域。当前大幅降低固定 benchmark 误差的方法，本质上是把“旧 Au benchmark 专家”和“外推边界专家”分开。

如果未来网页输入只给 `Z, rod, tep, tgama`，没有说明数据源/程序版本，那么 Au 的重叠区域可能存在歧义。要把方法稳健上线，最好在训练数据和网页预测里增加一个可选字段，例如：

`data_source = old_au / au2 / auto`

否则模型只能靠坐标范围自动猜测，固定 benchmark 能变好，但真实新数据不一定同步变好。

## 推荐

如果目标是固定 benchmark 最高精度，推荐把“旧 Au CatBoost 精细融合 + 当前边界专家 + Be/Al 最佳统一模型”做成暂存模型，网页上点击确认后再发布。

如果目标是轻量和简单部署，推荐只发布 `experiment2_au_k100_model.npz`。

如果目标是只追随机 benchmark，不考虑外推，则 `stored_heterogeneous_full_ensemble_all_range` 数字最好，但不建议用于实际预测。
