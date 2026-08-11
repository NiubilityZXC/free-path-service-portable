# FLASH4.8 便携复制与排除清单

本说明只描述未上传 GitHub 的本机授权 FLASH 副本。授权原树约 52 GB；本机便携副本保留代码、配置、输入、构建工具、HYPRE/CUDA 扩展、项目结构、项目独立覆盖源码和现成可执行文件，整理后约 3.3 GB。

## 保留

- 根目录 `source/`、`bin/`、`lib/`、`sites/`、`tools/`、`docs/`、`setup`、`RELEASE*`
- `gpu/` 源码、测试/benchmark 源码、`deps/src/hypre-3.1.0` 和 `deps/install/hypre-cuda-3.1.0`
- 每个项目的 `flash.par*`、`setup_call`、`setup_units`、`setup_datafiles`、Makefile、Units、输入表、项目独立源码、说明和现成 `flash4*`
- 复制时保留硬链接和相对符号链接；便携副本断链数为 0

## 排除

- 678 个 `_hdf5_chk_*` 检查点，表观合计约 44.63 GB
- 1,772 个 `_hdf5_plt_cnt_*` / `_hdf5_part_*` 绘图与粒子输出，表观合计约 13.13 GB
- 运行日志、`.out/.err/.pid`、`fort.*`、`.success`
- 约 52,020 个 `.o/.mod/.smod` 编译中间文件（约 859 MB）
- `.git`、pytest/Python 缓存和 `gpu/deps/build/`（约 225 MB，可由源码重建）

表观统计会因原树硬链接重复计数而高于 `du` 的 52 GB，不应直接相加。

## 已保留的历史二进制

原树有大量 CPU/GPU 实验版本。为满足“代码和可运行资产放在一起”，便携包仍保留这些 `flash4*`；正式推荐仅使用 CPU 稳定版、`gpu-unified-v19` 或 EOS 回退 `gpu-fused-v8`，不要使用已知失败的 v6。

## 未携带检查点的项目

`Al_shell_OR1cm_t4p5um_15MA_150ns_AMR` 当前主配置要求第 17 号检查点。该配置作为历史状态保留，但对应大型 checkpoint 被排除；从头运行前需关闭 restart，续跑则需单独补回检查点。
