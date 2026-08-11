# 统一自由程模型方法报告

## 1. 固定数据格式

后续训练、评测、预测都使用同一套标准格式，不再在网页或命令行选择数据集、坐标模式或特征。

训练与评测文件固定为 5 列：

```text
Z rod tep tgama lnu
```

预测输入固定为 4 列：

```text
Z rod tep tgama
```

当前约定：

| Z | 元素 | 来源 | 坐标处理 |
|---:|---|---|---|
| 4 | Be | `data_Be.txt` | 原前三列已经是标准坐标，直接使用 |
| 13 | Al | `data_Al.txt` | 原前三列已经是标准坐标，直接使用 |
| 79 | Au | `data.txt` | 原前三列先取 `log10` 后进入标准坐标 |

标准训练文件已经生成：

```text
free_path_model_outputs/unified_standard_training_data.txt
```

未来继续训练时，新数据也按这 5 列追加或另存为同格式文件。`lnu` 必须为正数。

## 2. 当前数据处理

| 来源 | 原始样本数 | 标准化后样本数 | 处理 |
|---|---:|---:|---|
| `data_Be.txt` | 125000 | 125000 | 写成 `Z=4`，保持原 `rod/tep/tgama` |
| `data_Al.txt` | 125000 | 125000 | 写成 `Z=13`，保持原 `rod/tep/tgama` |
| `data.txt` | 125000 | 124913 | 写成 `Z=79`，`rod/tep/tgama` 转为 `log10`，非正 `lnu` 行不参与训练 |

当前 `data.txt` 已删除 87 行非正 `lnu`，当前文件最小 `lnu=1.337e-08`。标准训练文件 `unified_standard_training_data.txt` 总行数为 374913，且没有非正 `lnu`。模型目标统一为 `log10(lnu)`，预测后再还原为 `lnu`。

## 3. 永久 Benchmark

当前有两套永久 benchmark，后续训练都不会使用这些点。

第一套是分散随机 benchmark，用于测试训练分布内的插值/常规泛化能力。划分规则是对标准行键：

```text
Z|rod|tep|tgama
```

做稳定 SHA256 哈希，约 10% 进入 benchmark。

第二套是两端连续外推 benchmark，用于测试范围外预测能力。划分规则是：对每个已训练元素，分别取 `rod` 最小的 10% 网格层和 `rod` 最大的 10% 网格层，形成低 `rod` 端和高 `rod` 端两段连续区域。这些点完全不参与训练，因此在 `rod` 方向属于两端边界外推测试。

当前划分：

| 集合 | 样本数 |
|---|---:|
| 训练 | 269748 |
| 随机 benchmark | 30190 |
| 两端连续外推 benchmark | 74975 |
| Z_4 两端连续外推 | 25000 |
| Z_13 两端连续外推 | 25000 |
| Z_79 两端连续外推 | 24975 |

文件：

```text
free_path_model_outputs/unified_permanent_benchmark_keys.txt
free_path_model_outputs/unified_permanent_benchmark_data.npz
free_path_model_outputs/unified_permanent_extrapolation_keys.txt
free_path_model_outputs/unified_permanent_extrapolation_data.npz
```

后续继续训练时，训练程序默认复用这些永久测试文件，并自动跳过所有 key 命中的行。只有显式使用 `--force-recreate-benchmark` 时才会重新划分测试集；网页继续训练入口不会使用这个参数。

当前正式模型元数据中记录了训练审计：

| 训练文件 | 总行数 | 参与训练 | 随机 benchmark 跳过 | 两端连续外推 benchmark 跳过 |
|---|---:|---:|---:|---:|
| `unified_standard_training_data` | 374913 | 269748 | 30190 | 74975 |

网页现在支持直接导入/导出训练数据：

| 操作 | 说明 |
|---|---|
| 导出当前训练数据 | 下载当前标准训练文件，默认是 `free_path_model_outputs/unified_standard_training_data.txt` |
| 追加合并 | 上传或粘贴 `Z rod tep tgama lnu` 五列数据，和现有训练文件合并，并按 `Z|rod|tep|tgama` 去重 |
| 覆盖替换 | 用上传或粘贴的数据替换目标训练文件，写入前自动备份旧文件 |
| 格式校验 | 导入时检查必须是 5 列数值，且 `lnu > 0` |

导入训练数据不会更新线上模型权重。导入后需要点击“开始暂存训练”，训练完成后再点击“确认更新模型权重”。后续训练仍会自动跳过永久随机 benchmark 和两端连续外推 benchmark 中的点。

## 4. 选定模型

基础模型使用一个整体模型：

```text
KDTree + 局部加权三次多项式回归
```

参数：

| 参数 | 值 |
|---|---:|
| `k_neighbors` | 520 |
| `local_degree` | 3 |
| `ridge_alpha` | 1e-3 |
| `distance_power` | 1.6 |
| `z_weight` | 0.5 |

`Z` 是标准输入格式的一列，和 `rod/tep/tgama` 一起参与归一化和近邻搜索。

当前网页线上预测在基础模型之外增加了一个高 Z 专用路由：

```text
Z=79/Au 训练盒内 -> 7 成员混合集成
其他元素或 Au 训练盒外 -> 统一 KDTree 局部回归模型
```

高 Z 集成模型只用于 `Z=79` 且 `rod/tep/tgama` 落在 Au 训练盒内的点；连续外推测试两端仍回落到统一模型，避免让专用模型在完全未覆盖的低/高 `rod` 区间乱外推。

当前高 Z 集成方法：

| 项 | 值 |
|---|---|
| 成员数 | 7 |
| 成员模型 | 4 个 CatBoostRegressor、2 个 ExtraTreesRegressor、1 个 HistGradientBoostingRegressor |
| 输入 | `rod, tep, tgama` |
| 目标 | `log10(lnu)` |
| 集成 | 固定加权融合：CatBoost 单模型、CatBoost 中位数、ExtraTrees、HGB |
| 训练行数 | 89741 |
| 训练耗时 | 251.457 s |

高 Z 优化中额外测试过但未采用的方案包括：更深 CatBoost、CatBoost `Depthwise/Lossguide`、LightGBM、XGBoost、ExtraTrees 单模型、HGB 单模型、KNN、局部多项式、RBF 插值、MLP 神经网络、随机 Fourier/RBF 核岭、邻域候选排序模型。当前混合集成是这些方案中在永久随机 benchmark 上表现最好的可上线方案。

Au 仍明显难于 Be/Al：最差误差点多为局部孤立深谷或尖峰，最近训练邻居的 `log10(lnu)` 往往与真值差 0.5 到 1.5 个数量级。一个不可上线的 oracle 检查显示，如果能在最近 128 个训练邻居中直接选到最接近真值的邻居，Au SMAPE 可到约 0.526%；但真实模型无法访问 benchmark 真值，因此不能可靠做出这种选择。要进一步接近 Be/Al 的 0.x% 水平，最有效的方向不是继续堆模型，而是在这些局部高曲率区域补充真实计算点或提高原始网格分辨率。

## 5. Benchmark 结果

当前推荐暂存模型：

```text
free_path_model_outputs/staged_unified_free_path_model.npz
```

该模型需要在网页点击“确认更新模型权重”后才会替换线上 active 权重。

随机 benchmark：

| element | 样本数 | SMAPE | log10 MAE | P90 倍数误差 | P99 倍数误差 |
|---|---:|---:|---:|---:|---:|
| overall | 30190 | 2.179057% | 0.00961875 | 1.056889 | 1.306942 |
| Z_4 | 10034 | 0.308956% | 0.00134179 | 1.008038 | 1.017684 |
| Z_13 | 9959 | 0.627487% | 0.00272556 | 1.015441 | 1.065465 |
| Z_79 | 10197 | 5.534622% | 0.02449569 | 1.142464 | 1.601178 |

两端连续外推 benchmark：

| element | 样本数 | SMAPE | log10 MAE | P90 倍数误差 | P99 倍数误差 |
|---|---:|---:|---:|---:|---:|
| overall | 74975 | 9.654967% | 0.04524218 | 1.218620 | 3.724671 |
| Z_4 | 25000 | 6.169194% | 0.02687211 | 1.161981 | 1.423750 |
| Z_13 | 25000 | 10.637207% | 0.05347862 | 1.202227 | 10.465422 |
| Z_79 | 24975 | 12.161006% | 0.05538596 | 1.336325 | 3.637597 |

## 6. 指标含义

| 指标 | 含义 | 越小越好吗 |
|---|---|---|
| `n` | 参与评测的样本数 | 不是误差指标 |
| `log10_mae` | `log10(lnu)` 的平均绝对误差。例如 0.03 大约对应典型倍数误差 `10^0.03≈1.07` | 是 |
| `log10_rmse` | `log10(lnu)` 的均方根误差，对少量大误差更敏感 | 是 |
| `mape_percent` | 平均相对误差百分比，`abs(pred-true)/abs(true)` 的平均值；当真实值很接近 0 时会被极端点放大 | 是 |
| `smape_percent` | 对称平均相对误差，`2*abs(pred-true)/(abs(pred)+abs(true))` 的平均值，比 MAPE 更适合跨数量级数据 | 是 |
| `median_factor_error` | 中位数倍数误差，1.01 表示一半样本误差小于约 1% | 越接近 1 越好 |
| `p90_factor_error` | 90% 样本的倍数误差不超过该值，例如 1.20 表示 90% 样本在 1.20 倍以内 | 越接近 1 越好 |
| `p99_factor_error` | 99% 样本的倍数误差不超过该值，用来看尾部误差 | 越接近 1 越好 |
| `max_factor_error` | 最大倍数误差，容易被极少数异常点影响 | 越接近 1 越好 |

默认更建议看 `SMAPE`、`log10_mae`、`P90/P99 倍数误差`。这些指标更适合自由程这种跨数量级目标。

## 7. 仿真可用阈值

网页中已经固定显示以下阈值，并自动给出“达标/未达标”判断。

| 范围 | SMAPE | log10 MAE | P90 倍数误差 | P99 倍数误差 |
|---|---:|---:|---:|---:|
| 整体默认 benchmark | ≤ 5.0% | ≤ 0.025 | ≤ 1.20 | ≤ 1.80 |
| 单元素 / 外部仿真文件 | ≤ 8.0% | ≤ 0.040 | ≤ 1.25 | ≤ 2.00 |

解释：

- 整体 benchmark 阈值用于判断当前线上模型总体是否可作为仿真替代。
- 单元素阈值用于判断某个元素或某个新仿真文件是否可直接用于工程预测。
- 网页现场评测入口现在只允许当前模型已经训练过的元素。

当前随机 benchmark 已达标；两端连续外推 benchmark 更严格，整体仍未达标：

| benchmark | element | SMAPE | log10 MAE | P90 倍数误差 | P99 倍数误差 | 结论 |
|---|---|---:|---:|---:|---:|---|
| 随机 | overall | 2.179057% | 0.00961875 | 1.056889 | 1.306942 | 达标 |
| 随机 | Z_4 | 0.308956% | 0.00134179 | 1.008038 | 1.017684 | 达标 |
| 随机 | Z_13 | 0.627487% | 0.00272556 | 1.015441 | 1.065465 | 达标 |
| 随机 | Z_79 | 5.534622% | 0.02449569 | 1.142464 | 1.601178 | 达标 |
| 两端连续外推 | overall | 9.654967% | 0.04524218 | 1.218620 | 3.724671 | 未达标 |
| 两端连续外推 | Z_4 | 6.169194% | 0.02687211 | 1.161981 | 1.423750 | 达标 |
| 两端连续外推 | Z_13 | 10.637207% | 0.05347862 | 1.202227 | 10.465422 | 未达标 |
| 两端连续外推 | Z_79 | 12.161006% | 0.05538596 | 1.336325 | 3.637597 | 未达标 |

网页指标表头和评测卡片中的 `?` 标记可以悬停查看公式和例子，包括 `SMAPE`、`log10 MAE`、`P90/P99 倍数误差`。

预测结果框还会显示单点审计：

| 字段 | 含义 |
|---|---|
| 数据命中 | 当前 `Z rod tep tgama` 是否精确存在于标准数据，以及它属于训练集、随机 benchmark、两端连续外推 benchmark，还是标准数据中不存在 |
| 预测耗时 | 当前模型完成一次预测的耗时 |
| 不透明度程序真值 | 勾选“同时调用真实程序”后，网页会调用当前目录代码编译出的 `SNOP_op_Ross_single` 单点程序现场计算真值 |
| 预测/真值误差 | 真实程序完成后显示相对误差、SMAPE、log10 误差对应的倍数误差 |
| 用时对比 | 显示模型预测耗时和真实程序运行耗时，并给出速度倍数 |

当前已经从 `SNOP_op_Ross3.F` 拆出了单点主程序：

```text
SNOP_op_Ross_single.F
build_single_point_simulators.py
free_path_model_outputs/simulators/Z_13/SNOP_op_Ross_single
free_path_model_outputs/simulators/Z_79/SNOP_op_Ross_single
```

网页调用真实程序时，会先把标准输入坐标还原为程序物理输入：

```text
rod_physical   = 10^rod
tep_physical   = 10^tep
tgama_physical = 10^tgama
```

然后调用对应元素的单点可执行文件。当前程序目录中已构建的真实程序元素包括 `Z=4,6,13,22,26,29,42,50,56,63,74,79,82,92`。网页现场评测默认只对已训练元素显示评测结果；当前训练元素是 `Z=4,13,79`。

## 8. 未训练过的原子能否预测

可以让统一基础模型对未训练过的 `Z` 输出预测值，因为 `Z` 是连续输入特征。但当前训练数据只有三个元素：Be (`Z=4`)、Al (`Z=13`) 和 Au (`Z=79`)。因此：

- 对 `4 < Z < 79` 且不等于 13 的元素，属于跨原子序数插值，只能作为趋势估计。
- 对 `Z < 4` 或 `Z > 79` 的元素，属于原子序数外推，风险更高。
- 当前没有“留一元素 benchmark”，因为只有三个元素，仍不足以可靠评估未见元素泛化能力。
- 如果后续增加更多元素数据，尤其是多个 Z 分布在 13 到 79 之间及两侧，可以用“按元素留出测试”评估真正的未见原子泛化能力。

网页预测入口会在输入未训练过的 Z 时给出提示，但仍会返回预测结果。现场评测和上传评测入口已限制为已训练元素；当前允许 `Z=4,13,79`。

## 9. 网页功能

网页服务：

```text
http://192.168.110.64:8790/
http://127.0.0.1:8790/
```

网页现在只接受固定格式：

- 预测：输入 `Z rod tep tgama`。
- 上传评测：每行 `Z rod tep tgama lnu`，只接受已训练元素。
- 训练元素 Benchmark：输入已训练 `Z`、仿真数据文件路径和抽样行数，网页会用最新线上模型评测。
- 继续训练：选择一个标准 5 列训练文件路径。
- 暂存训练完成后，必须点击“确认更新模型权重”才会替换线上模型。

## 10. 命令行

重新训练正式模型：

```bash
cd /home/user/Rosseland_opa_new_hetero/equally
python train_unified_free_path_model.py --standard-data free_path_model_outputs/unified_standard_training_data.txt
```

预测：

```bash
python predict_unified_free_path.py 13 -1.870 1.771 1.755
python predict_unified_free_path.py 79 2.7781513213184033 3.6020631833806736 2.778158343802252
```

测试：

```bash
python -m unittest test_unified_free_path_model.py test_free_path_model.py test_two_dataset_free_path_model.py
```
