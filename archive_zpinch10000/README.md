# zpinch10000

这个目录保存了当前阶段“两个 sheet 合并训练”的代码、数据和结果。

## 主要文件

- `train_all_data_combined_xlsx.py`
  - 两个 sheet 合并训练的主脚本。
- `train_all_data_combined_xlsx.ipynb`
  - Jupyter 版本。
  - 包含详细中文注释、固定划分训练流程、8 个模型的统一测试结果和训练损失折线图。
- `train_all_data_combined_xlsx_standalone.ipynb`
  - 完全独立版 Jupyter。
  - 不再依赖项目里的其它 `.py` 文件。
  - 只需要当前目录下的 `零维表格转换 (2).xlsx` 和 `ai` 环境里的库即可运行。
- `零维表格转换 (2).xlsx`
  - 原始 Excel 数据。
- `ai_xlsx_all_in_training_outputs/`
  - 合并训练后的结果目录。
- `合并训练结果解读.md`
  - 对代码、文件、指标和结果的中文解读。

## 如何使用

### 1. 打开 Notebook

推荐直接打开：

- `train_all_data_combined_xlsx.ipynb`
- `train_all_data_combined_xlsx_standalone.ipynb`

Notebook 里已经包含：

- 两个 sheet 的合并训练流程
- 详细中文注释
- `PowerLaw / Poly2Ridge / KNN / MLP / RandomForest / LightGBM / XGBoost / CatBoost` 共 8 个模型
- 固定测试集上的模型指标表
- `MLP / LightGBM / XGBoost / CatBoost` 的训练损失折线图
- 对完整汇总结果 `summary_metrics.json` 的读取展示

如果你要一个“拿走 ipynb 和 xlsx 就能单独跑”的版本，优先用：

- `train_all_data_combined_xlsx_standalone.ipynb`

### 2. 直接运行脚本

如果要直接跑主脚本：

```bash
/home/user/miniconda3/bin/conda run -n ai python /home/user/zpinch10000/train_all_data_combined_xlsx.py
```

### 3. 执行 Notebook

如果要从命令行执行 Notebook：

```bash
/home/user/miniconda3/envs/ai/bin/jupyter nbconvert --to notebook --execute --inplace /home/user/zpinch10000/train_all_data_combined_xlsx.ipynb
```

独立版命令：

```bash
/home/user/miniconda3/envs/ai/bin/jupyter nbconvert --to notebook --execute --inplace /home/user/zpinch10000/train_all_data_combined_xlsx_standalone.ipynb
```

## 最重要的结果文件

- `ai_xlsx_all_in_training_outputs/summary_metrics.json`
  - 最完整的结果汇总。
- `ai_xlsx_all_in_training_outputs/combined_fixed_split_metrics_notebook.csv`
  - Notebook 固定测试集上的 8 模型结果表。
- `ai_xlsx_all_in_training_outputs/combined_fixed_split_source_target_metrics_notebook.csv`
  - Notebook 固定测试集按 `source / target` 拆分后的结果表。
- `ai_xlsx_all_in_training_outputs/combined_random_split_metrics.csv`
  - 合并数据后的整体随机划分指标。
- `ai_xlsx_all_in_training_outputs/combined_random_split_target_metrics.csv`
  - 合并后测试集中，来自目标 sheet 的子集指标。
- `ai_xlsx_all_in_training_outputs/fig_training_loss_curves.png`
  - 训练损失折线图。
- `ai_xlsx_all_in_training_outputs/fig_combined_true_vs_pred.png`
  - 最佳模型在固定测试集上的真实值-预测值散点图。

## 当前结论

- 合并后最佳模型仍然是 `RandomForest`。
- 目标域测试误差相比“只用 source 训练再测 target”有明显下降。
- 但 `mass holdout` 结果仍然说明：质量维度的强外推能力还不够。
- 目前最合理的结论是：
  - 目标域内插值能力已经显著增强
  - 对未见质量值或未见工况的强泛化仍需补点和更针对性的建模
