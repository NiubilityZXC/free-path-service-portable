# 未携带的大型/临时资产

原自由程工程约 18 GB。便携包保留当前正式模型、正式训练数据、永久测试、全部本地 Python/Fortran/CUDA 代码和可用真实程序；以下内容因不参与当前服务或属于可再生临时结果而未复制。

## 停用高 Z 与模型搜索产物

- `free_path_model_outputs/experiment*`：约 10.739 GiB
- 当前旧 `high_z_*` CatBoost/joblib 集成：约 6.058 GiB
- `staged*` 临时发布产物：约 0.608 GiB
- `tmp_model_search/`、`catboost_info/`、各类 `*bak*`、`*backup*`、`*before*`、`*changed_by*`

排除依据：当前生产元数据明确为 `route_enabled=false`，网页预测全部走 `unified_free_path_model.npz`。原实现虽关闭路由仍预加载数 GB 权重；便携副本已修正为看到禁用元数据就不加载，也不要求训练发布时复制占位高 Z 模型。

高 Z 训练源码 `train_high_z_free_path_model.py` 及相关研究脚本仍在包内；需要重新启用这条实验路线时，安装完整训练依赖并重新训练即可。

## 缓存、日志与可再生文件

- `.git/`、`__pycache__/`、`*.pyc`
- 浏览器 `node_modules/`、Playwright 临时快照和临时测试输出；`output/playwright/` 中 README 使用的三张实际页面截图保留
- 服务日志、PID、时间/状态文件
- 便携副本创建时可重新构建的部分缓存和备份

## 兼容性说明

- 旧 `predict_free_path.py`、`predict_two_dataset_free_path.py` 等研究入口源码保留，但它们的旧紧凑模型不是当前 8790 服务资产；当前正式入口是统一模型。
- 已编译 Fortran/CUDA 二进制具有平台依赖，不是跨操作系统文件。源码、Cuba 库和低 Z 源归档已保留，可在匹配工具链上重编。
- 仅本机授权副本把 52 GB FLASH 原树中的核心源码、工具、项目配置与现成可执行文件整理为约 3.3 GB；该副本不上传 GitHub。排除的约 48.7 GB 主要是 HDF5 plot/checkpoint、日志、PID、缓存和可再生 `.o/.mod`；详见 `FLASH_PORTABLE_EXCLUSIONS.md`。
