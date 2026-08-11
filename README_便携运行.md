# 自由程服务完整便携包

本目录有两种形态：本机便携副本可另外保留合法取得的 `FLASH4.8/`；GitHub 克隆则不含 FLASH 本体。两者都包含当前正式自由程服务、统一模型、模型训练/推理代码、永久测试数据、真实单点程序源码、电容器评估、当前 Z-pinch 模型和历史 Z-pinch 整理包。

便携包根目录：克隆或复制后的 `free_path_service_portable/`。GitHub 图文入口见 `README.md`。

> **许可提醒：** 本地便携目录保留 `FLASH4.8/`，但它使用独立受限许可证，已被 Git 忽略且不会上传 GitHub。

## 最快启动：Docker（推荐）

目标电脑安装 Docker Desktop 或 Docker Engine + Compose 后：

```bash
cd free_path_service_portable
docker compose up --build -d
docker compose ps
```

访问：

- 主页面：`http://目标电脑IP:8790/`
- 独立电容页面：`http://目标电脑IP:8890/`

停止与查看日志：

```bash
docker compose down
docker compose logs -f freepath
docker compose logs -f capacitor
```

Compose 设置了 `restart: unless-stopped`，电脑重启后 Docker 会自动恢复服务。第一次构建需要联网下载 Python 镜像和依赖。模型推理不要求 GPU。

Docker 构建上下文会跳过体积较大的 `FLASH4.8/` 和历史 Z-pinch 归档；Compose 运行时会把整个工作区挂载到 `/app`；只有宿主机另行合法放入 `FLASH4.8/` 时，容器内才可访问 FLASH 源码和项目。

## 原生 Linux 启动

验证环境为 Linux x86_64、Python 3.13.12。目标电脑先安装 Python 3.13 及 venv，然后：

```bash
cd free_path_service_portable
./setup.sh
./start_all.sh
./status.sh
```

停止：

```bash
./stop_all.sh
```

离线模型自检：

```bash
.venv/bin/python scripts/portable_smoke_test.py
```

完整研究/训练依赖：

```bash
./setup.sh --full
```

脉冲电容的深度学习重训及历史 MLP 还需要 PyTorch：

```bash
./setup.sh --full --torch
```

如果目标电脑不是 Python 3.13，建议直接用 Docker。模型 pickle 来自 NumPy 2.x / scikit-learn 1.8.0 环境；随意降级依赖可能无法加载或产生不同结果。

## 当前正式代码与模型

### 自由程主服务

- 网页/API：`freepath_service/free_path_web_app.py`
- 当前推理核心：`freepath_service/unified_free_path_core.py`
- 当前模型：`freepath_service/free_path_model_outputs/unified_free_path_model.npz`
- 当前训练：`freepath_service/train_unified_free_path_model.py`
- 当前训练数据：`freepath_service/free_path_model_outputs/unified_standard_training_data.txt`
- 永久 benchmark / extrapolation 数据、键和指标：同一 `free_path_model_outputs/` 目录
- 高 Z 专用路由当前明确关闭，所有正式请求由统一模型处理

训练参数建议从网页“训练与验证”页操作；命令行可先查看：

```bash
.venv/bin/python freepath_service/train_unified_free_path_model.py --help
```

### 当前 Z-pinch

```bash
.venv/bin/python zpinch/predict_current.py --I 50 --tr 300 --liner-r 5 --foam-r 0.6 --m 30 --Z 13
.venv/bin/python zpinch/optimize_zpinch_velocity_first.py
```

当前模型是 velocity-first 链路。随包提供 2730 行标准 CSV，可在另一台电脑重训；原始 DOC 也保留作溯源。

### 脉冲电容器

- 服务：`pulse_capacitor_online_eval/app.py`
- 训练：`pulse_capacitor_online_eval/tools/train_strict_deep_models.py`
- 模型实现：`pulse_capacitor_online_eval/deep_learning/strict_waveform_models.py`
- 原始 CSV、转换工具、严格模型报告和专利源文件均已随包

### FLASH 和真实单点程序

- `Cuba-4.2.2/`、`Rosseland_opa_low_Z.zip`、Fortran/CUDA 源码和已编译模拟器已随包。
- 已编译真实单点程序只保证兼容 Linux x86_64、CUDA 13、libgfortran5、LAPACK、OpenBLAS 和匹配的 NVIDIA 运行环境；普通电脑仍可使用模型推理，但真实程序按钮可能不可用。
- 重编时可设置 `CUDA_HOME=/你的/cuda/目录`。
- 仅本机授权便携副本中的 `FLASH4.8/` 保留核心源码、构建工具、站点配置、GPU 扩展、项目配置和现成 `flash4` 可执行文件；GitHub 仓库明确不包含这些内容。
- 本机 FLASH 副本中的 73,285 个相对链接已修复为包内目标并验证无断链；GitHub 克隆无需也无法执行这项 FLASH 链接检查。详情见 `FLASH_PORTABLE_EXCLUSIONS.md`。

换机器后使用 FLASH 前先加载便携环境；预编译程序只有在系统 ABI 匹配时才能直接使用，否则按说明重编：

```bash
source scripts/flash_env.sh
.venv/bin/python scripts/rebuild_flash_project.py improved_MRT --dry-run
.venv/bin/python scripts/repair_flash_symlinks.py
```

## 为什么原来 8790 会停止

原服务是手工 `nohup/setsid` 启动，没有 systemd、Docker restart policy 或其他服务管理器；主机重启后不会自动回来。便携包提供了受控 PID 的原生启动/停止脚本，Docker Compose 方案还提供重启自动恢复。

## 搬运

可以直接复制整个 `free_path_service_portable/` 文件夹，也可以生成带 SHA256 的压缩包。GitHub 私有仓库仅同步应用、模型、训练/推理代码和数据，不同步受限的 `FLASH4.8/` 本体：

```bash
./pack_archive.sh
```

目标电脑解压后，优先按 Docker 方式启动。完整文件清单见 `PORTABLE_CONTENTS.md`，未携带的大型无效资产见 `EXCLUDED_LARGE_ARTIFACTS.md`，逐文件哈希见 `SHA256SUMS`。
