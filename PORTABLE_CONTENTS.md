# 便携包内容清单

## 运行控制

- `setup.sh`：新建 `.venv` 并安装锁定依赖
- `start_all.sh`：启动 8790 主服务和 8890 电容服务
- `stop_all.sh`：核对进程命令行后安全停止
- `status.sh`：显示 PID 与 HTTP 状态
- `pack_archive.sh`：生成可搬运 tar.gz 和归档 SHA256
- `Dockerfile`、`compose.yaml`：容器启动与重启恢复
- `scripts/wait_http.py`：HTTP 就绪检查
- `scripts/portable_smoke_test.py`：统一模型、Z-pinch、数据资产离线自检
- `scripts/flash_env.sh`：设置便携 FLASH/CUDA/HYPRE 环境
- `scripts/rebuild_flash_project.py`：根据项目 `setup_call` 重做 setup 和编译
- `scripts/repair_flash_symlinks.py`：审计/修复旧机器 FLASH 符号链接
- `scripts/regenerate_manifests.sh`：重建本地链接清单和 GitHub 文件 SHA256
- `scripts/generate_readme_model_figures.py`：从正式 artifact 与指标报告复算 README 模型图

## 文档图表与页面证据

- `output/playwright/`：8790、8890 与 Z-pinch 当前页面实拍
- `output/model-docs/`：自由程、Z-pinch、脉冲电容架构效果章节的五张可复现图

## `freepath_service/`：当前主服务

包含主网页/API、当前与历史训练/推理 Python 源码、Fortran/CUDA 真实程序源码、数据表、构建脚本、测试、报告和说明。

当前正式资产：

- `free_path_model_outputs/unified_free_path_model.npz`
- `free_path_model_outputs/unified_free_path_metrics.json`
- `free_path_model_outputs/unified_standard_training_data.txt`
- `free_path_model_outputs/unified_permanent_benchmark_*`
- `free_path_model_outputs/unified_permanent_extrapolation_*`
- `free_path_model_outputs/unified_free_path_benchmark_metrics.csv`
- `free_path_model_outputs/unified_free_path_extrapolation_metrics.csv`
- `free_path_model_outputs/simulators/`

当前高 Z 路由元数据是 `route_enabled=false`，因此运行时不加载 CatBoost/joblib 历史高 Z 集成权重。

## `pulse_capacitor_online_eval/`：脉冲电容器

包含标准库 Web 服务、前端、原始 CSV、转换与训练工具、深度模型实现、严格评估报告、测试和原专利 DOCX。正式服务不依赖保存的神经网络权重；严格结果和曲线保存在报告 JSON，训练脚本会重新训练模型。

## `zpinch/`：当前在线 Z-pinch

包含当前网页实际使用的 14 维特征定义、velocity-first 优化训练、正式 pickle、摘要、图表、原始 DOC、2730 行规范 CSV、Notebook、报告和独立推理入口。

## `archive_zpinch10000/`：历史整理包

约 311 MB，保留旧“两 sheet 合并训练”的 RandomForest/XGBoost/CatBoost 等代码、权重、结果和中文解读。它不是当前网页模型，不能替代 `zpinch/` 中的 velocity-first 链路。

## 真实程序与 FLASH 工具

- `Cuba-4.2.2/`：已构建 `libcuba.a` 及头文件
- `Rosseland_opa_low_Z.zip`：低 Z 真实程序源归档
- `cli-anything-flash/`：FLASH 自然语言/CLI harness 代码
- `FLASH4.8/`（仅本地，不上传 GitHub）：约 3.3 GB 的授权 FLASH4.8 核心源码、工具、GPU 扩展、全部项目配置、73,285 个包内相对链接和 45 个现成 `flash4` 可执行文件；大型运行输出已排除

## 依赖锁

- `requirements-runtime.txt`：当前服务、统一模型、Z-pinch 推理和 FLASH CLI
- `requirements-training.txt`：高 Z 研究训练与表格模型额外依赖
- `requirements-torch.txt`：电容深度学习和历史 MLP 可选依赖
