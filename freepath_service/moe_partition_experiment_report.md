# MoE / 物理状态分区离线实验报告

- 生成时间：2026-07-03 14:22:20
- 训练数据：`/home/user/Rosseland_opa_new_hetero/equally/free_path_model_outputs/experiment_standard_with_old_au_and_au2.txt`
- 固定随机 benchmark：`/home/user/Rosseland_opa_new_hetero/equally/free_path_model_outputs/unified_permanent_benchmark_data.npz`
- 固定连续外推 benchmark：`/home/user/Rosseland_opa_new_hetero/equally/free_path_model_outputs/unified_permanent_extrapolation_data.npz`
- 本实验没有发布模型，也没有修改网页正在使用的权重。

## 随机 benchmark 排名

| 排名 | 方法 | overall SMAPE | Au SMAPE | Au log10 MAE | Au P99倍数 | 路由行数 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `stored_heterogeneous_full_ensemble_all_range` | 2.203235% | 5.874243% | 0.02600647 | 1.636384 | -1 |
| 2 | `stored_heterogeneous_full_ensemble_midrod` | 2.211456% | 5.898581% | 0.02611312 | 1.636384 | 9988 |
| 3 | `stored_catboost_single_midrod` | 2.341496% | 6.283588% | 0.02781420 | 1.645289 | 9988 |
| 4 | `catboost_direct_sampled+all_au` | 2.513009% | 6.791383% | 0.03036974 | 1.856151 | 10197 |
| 5 | `catboost_direct_sampled+ood_distance_q95_0.0711` | 2.513120% | 6.791711% | 0.03037119 | 1.856151 | 10194 |
| 6 | `catboost_direct_sampled+inner_rod_p10_p95` | 2.536702% | 6.861528% | 0.03068012 | 1.855990 | 9451 |
| 7 | `feature_kmeans_hard_hgb+ood_distance_q95_0.0711` | 2.564604% | 6.944138% | 0.03100717 | 1.830766 | 10194 |
| 8 | `feature_kmeans_hard_hgb+all_au` | 2.565127% | 6.945687% | 0.03101391 | 1.830766 | 10197 |
| 9 | `feature_kmeans_hard_hgb+inner_rod_p10_p95` | 2.583273% | 6.999410% | 0.03125319 | 1.830766 | 9451 |
| 10 | `input_output_kmeans_soft_classifier_hgb+ood_distance_q95_0.0711` | 2.586929% | 7.010234% | 0.03130326 | 1.822349 | 10194 |
| 11 | `input_output_kmeans_soft_classifier_hgb+all_au` | 2.587467% | 7.011828% | 0.03131021 | 1.822349 | 10197 |
| 12 | `input_output_kmeans_soft_classifier_hgb+inner_rod_p10_p95` | 2.606671% | 7.068686% | 0.03155801 | 1.822349 | 9451 |
| 13 | `residual_hgb_on_unified+all_au` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 10197 |
| 14 | `residual_hgb_on_unified+ood_distance_q95_0.0711` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 10194 |
| 15 | `validation_nnls_soft_stack+all_au` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 10197 |
| 16 | `validation_nnls_soft_stack+ood_distance_q95_0.0711` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 10194 |
| 17 | `residual_hgb_on_unified+inner_rod_p10_p95` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 9451 |
| 18 | `validation_nnls_soft_stack+inner_rod_p10_p95` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 9451 |
| 19 | `baseline_unified_local_k100` | 2.630416% | 7.138987% | 0.03190248 | 1.926742 | 0 |
| 20 | `hgb_global_direct+ood_distance_q95_0.0711` | 3.054705% | 8.395168% | 0.03741912 | 1.936915 | 10194 |

## 连续外推 benchmark 排名

| 排名 | 方法 | overall SMAPE | Au SMAPE | Au log10 MAE | Au P99倍数 | 路由行数 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `residual_hgb_on_unified+inner_rod_p10_p95` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 12500 |
| 2 | `validation_nnls_soft_stack+inner_rod_p10_p95` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 12500 |
| 3 | `baseline_unified_local_k100` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 0 |
| 4 | `stored_catboost_single_midrod` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 0 |
| 5 | `stored_heterogeneous_full_ensemble_midrod` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 0 |
| 6 | `residual_hgb_on_unified+ood_distance_q95_0.0711` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 23611 |
| 7 | `validation_nnls_soft_stack+ood_distance_q95_0.0711` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 23611 |
| 8 | `residual_hgb_on_unified+all_au` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 24975 |
| 9 | `validation_nnls_soft_stack+all_au` | 7.783151% | 8.069213% | 0.03683382 | 2.952043 | 24975 |
| 10 | `stored_heterogeneous_full_ensemble_all_range` | 8.728602% | 10.907459% | 0.04923523 | 3.081886 | -1 |
| 11 | `catboost_direct_sampled+ood_distance_q95_0.0711` | 8.926577% | 11.501780% | 0.05169320 | 3.048597 | 23611 |
| 12 | `catboost_direct_sampled+inner_rod_p10_p95` | 8.974511% | 11.645678% | 0.05220056 | 2.748803 | 12500 |
| 13 | `catboost_direct_sampled+all_au` | 9.078963% | 11.959241% | 0.05367821 | 3.042552 | 24975 |
| 14 | `input_output_kmeans_soft_classifier_hgb+ood_distance_q95_0.0711` | 9.353724% | 12.784075% | 0.05746060 | 3.269707 | 23611 |
| 15 | `feature_kmeans_hard_hgb+ood_distance_q95_0.0711` | 9.431249% | 13.016805% | 0.05837825 | 3.097465 | 23611 |
| 16 | `input_output_kmeans_soft_classifier_hgb+inner_rod_p10_p95` | 9.448137% | 13.067505% | 0.05845097 | 2.749951 | 12500 |
| 17 | `input_output_kmeans_soft_classifier_hgb+all_au` | 9.573092% | 13.442618% | 0.06032861 | 3.277027 | 24975 |
| 18 | `feature_kmeans_hard_hgb+inner_rod_p10_p95` | 9.581034% | 13.466461% | 0.06022717 | 2.749951 | 12500 |
| 19 | `feature_kmeans_hard_hgb+all_au` | 9.701558% | 13.828275% | 0.06192125 | 3.097465 | 24975 |
| 20 | `hgb_global_direct+ood_distance_q95_0.0711` | 9.989860% | 14.693758% | 0.06594520 | 3.317565 | 23611 |

## 说明

- `all_au`：所有 Au 点都走专家模型。
- `inner_rod_p10_p95`：只在 Au 训练 rod 的 10%-95% 内启用专家，边界区域回退统一模型。
- `ood_distance_q95`：离训练样本太远就回退统一模型。
- `stored_*` 是前面已经训练出的高 Z CatBoost/异构集成候选，本次直接纳入同一张表对比。
- overall 里 Be/Al 使用同一个基础统一模型，因此对比主要反映 Au 专家和路由策略的差异。
