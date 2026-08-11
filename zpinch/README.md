# 当前 Z-pinch 模型

这是当前网页实际调用的 velocity-first 链路，不是根目录 `archive_zpinch10000/` 中的历史 energy-first/RandomForest 链路。

- 模型定义与 14 维特征顺序：`train_zpinch_surrogate.py`
- 当前优化训练：`optimize_zpinch_velocity_first.py`
- 当前权重：`ai_training_outputs/best_velocity_first_optimized_model_artifact.pkl`
- 当前摘要：`ai_training_outputs/velocity_first_optimized_summary.json`
- 独立推理：`predict_current.py`
- 可迁移训练数据：`source_data/zpinch_training_data.csv`（2730 行）
- 原始溯源文件：`source_data/*.doc`

示例：

```bash
../.venv/bin/python predict_current.py --I 50 --tr 300 --liner-r 5 --foam-r 0.6 --m 30 --Z 13
../.venv/bin/python optimize_zpinch_velocity_first.py
```

基准输出约为 `E=5.030381733204523 MJ/cm`、`v=-57910170.28239814 cm/s`。模型 pickle 绑定 `train_zpinch_surrogate` 顶层类/函数和 14 维特征顺序，不要改名或重排。
