# FLASH 自由程模型接入说明

## 已定位内容

- `user_flash` 不是系统登录名，而是账号备注名。实际用户是 `Tsouzou_von_Habsburg-Hohenschaw`，家目录是 `/home/user2`，实际指向 `/home/Von_Hohenschaws`。
- FLASH 树在 `/home/user2/Haigerloch/FLASH4.8`。
- 当前重点运行目录是 `/home/user2/Haigerloch/FLASH4.8/improved_MRT`，同类目录还有 `improved_MRT_AMR` 和 `improved_MRT_long`。
- 当前 `flash.par` 中 `useRadTrans = .false.` 且 `useOpacity = .false.`。如果不开启这两项，FLASH 当前不会真正调用 opacity/free-path 路径。
- 已直接修改并重新编译 `/home/user2/Haigerloch/FLASH4.8/improved_MRT/flash4`。原文件备份在：

```text
/home/user2/Haigerloch/FLASH4.8/improved_MRT/freepath_model_patch_backup_20260711_133513
/home/user2/Haigerloch/FLASH4.8/freepath_model_patch_backup_20260711_134835
```

## FLASH 当前查表路径

当前运行参数里 transport opacity 使用 IONMIX Rosseland 表：

```text
op_fillTrans = "op_tabro"
op_fillFileType = "ionmix4"
op_fillFileName = "DD-006-imx.cn4"

op_lineTrans = "op_tabro"
op_lineFileType = "ionmix4"
op_lineFileName = "Be-006-imx.cn4"

op_vacuTrans = "op_tabro"
op_vacuFileType = "ionmix4"
op_vacuFileName = "Be-006-imx.cn4"
```

调用链是：

```text
RadTrans_* -> Opacity(soln, ngrp, ...) -> op_getSpeciesTabulatedOpacities
           -> op_getSpeciesROTableOpacity
```

关键文件：

```text
/home/user2/Haigerloch/FLASH4.8/improved_MRT/Opacity.F90
/home/user2/Haigerloch/FLASH4.8/improved_MRT/op_getSpeciesTabulatedOpacities.F90
/home/user2/Haigerloch/FLASH4.8/improved_MRT/op_getSpeciesROTableOpacity.F90
```

`op_getSpeciesROTableOpacity` 原来做的是 `(speciesTemperature, speciesDensity, speciesEnergyGroup)` 上的双线性插值，返回 `opacityRO`，单位是 `cm^2/g`。这里的 `speciesDensity` 是离子数密度 `# ions/cm^3`，不是自由程模型需要的质量密度；所以接入模型时使用 `op_cellMassDensity * op_cellSpeciesMassFractions(species)` 得到该 species 的质量密度。

## 网页模型接口

网页服务地址：

```text
http://127.0.0.1:8790
```

FLASH 接入控制：

```bash
curl http://127.0.0.1:8790/api/flash/control
```

网页上的“FLASH 接入控制”栏会保存 `enabled`、`predictionMode` 和 `modelElements`。现在 FLASH helper 会读取这个接口：`enabled=false` 时回退原 IONMIX/查表；`enabled=true` 且元素被勾选时才使用网页模型。

网页还新增了单独的“FLASH 控制台”页签，用来图形化管理 FLASH 操作：

```text
运行目录：自动扫描 FLASH 树下带 flash.par/flash4 的项目，目前可见 8 个项目
新建项目：从现有运行目录复制出新项目，支持轻量复制或完整复制
运行状态：查看 flash4、flash.par、后台运行 pid、最近日志
全部参数：flash.par 中所有参数都会显示，可按分组/关键字搜索，布尔/数字/字符串分别用开关或输入框控制
新增参数：可在网页中新增合法参数名，保存时追加到 flash.par
运行控制：启动 ./flash4、停止网页启动的进程、执行 make -j2
命令行：在网页里对选中的 FLASH 运行目录执行命令
结果查看：直接查看 log/txt/dat/par/csv 和普通图片文件
结果可视化：FLASH HDF5 plot/checkpoint 文件可在网页选择变量后直接渲染二维 PNG 图像；DAT/CSV/TXT 数字表可选择 X/Y 列绘制曲线图。后端使用 h5py + matplotlib 读取和绘图
```

对应接口：

```text
GET  /api/flash/status
POST /api/flash/params
POST /api/flash/project
POST /api/flash/run
POST /api/flash/stop
POST /api/flash/command
GET  /api/flash/file
GET  /api/flash/plot
```

健康检查：

```bash
curl http://127.0.0.1:8790/health
```

单点预测：

```bash
curl -sS -H 'Content-Type: application/json' \
  -d '{"Z":13,"rod":0,"tep":3,"tgama":3}' \
  http://127.0.0.1:8790/api/predict
```

返回中最重要字段：

```text
lnu        Rosseland 自由程，单位按训练程序为 cm
log10_lnu  log10(lnu)
```

网页模型固定输入格式是：

```text
Z rod tep tgama
```

当前标准坐标含义：

```text
rod   = log10(rho[g/cm^3])
tep   = log10(Tele[eV])
tgama = log10(Trad[eV])
```

训练范围约为：

```text
rod   [-1.87, 4.477]    rho 约 1.35e-2 到 3.0e4 g/cm^3
tep   [1.771, 5.301]    Tele 约 59 到 2.0e5 eV
tgama [1.755, 4.477]    Trad 约 57 到 3.0e4 eV
```

注意：`improved_MRT/flash.par` 里 Be liner 初始温度是 0.5 eV，fill/vacuum 密度是 `1e-5 g/cm^3`，这些初值在模型训练范围外。若强行替换，要么只在范围内启用模型并在范围外回退 IONMIX 表，要么补充低温/低密度训练数据。

另一个限制是元素覆盖。当前网页模型直接训练过的元素是：

```text
Z = 4   Be
Z = 13  Al
Z = 79  Au
```

FLASH 当前 fill 是 DD fuel，等效 `Z=1`，不在训练元素里。因此第一版接入建议只对 Be liner/vacuum 的 `Z=4` 在训练范围内启用模型；DD fill 和所有超范围状态先回退原 IONMIX 表。

## 最小替换公式

FLASH 需要的是 transport opacity：

```text
opacityRO = kappa_R [cm^2/g]
```

网页模型给的是自由程：

```text
lnu [cm]
```

纯材料单元的换算是：

```text
kappa_R = 1 / (rho * lnu)
alpha   = rho * kappa_R = 1 / lnu
```

其中 `rho` 必须是该材料的质量密度 `g/cm^3`，不是 IONMIX 表插值函数当前收到的离子数密度。

## 已直接修改 FLASH

新增运行参数，默认全部保持原行为：

```text
opacity_useFreePathModel = .false.
opacity_freePathHelper = "python3 /home/user/Rosseland_opa_new_hetero/equally/flash_free_path_opacity_probe.py"
opacity_freePathUrl = "http://127.0.0.1:8790"
opacity_freePathTimeout = 5.0
```

改动位置：

```text
/home/user2/Haigerloch/FLASH4.8/source/physics/materialProperties/Opacity/OpacityMain/Multispecies/Opacity_data.F90
/home/user2/Haigerloch/FLASH4.8/source/physics/materialProperties/Opacity/OpacityMain/Multispecies/Opacity_init.F90
/home/user2/Haigerloch/FLASH4.8/source/physics/materialProperties/Opacity/OpacityMain/Multispecies/Config
/home/user2/Haigerloch/FLASH4.8/source/physics/materialProperties/Opacity/OpacityMain/Multispecies/method/Tabulated/op_getSpeciesROTableOpacity.F90
/home/user2/Haigerloch/FLASH4.8/improved_MRT/flash.par
/home/user2/Haigerloch/FLASH4.8/improved_MRT/rp_initParameters.F90
```

当前实现逻辑：

1. 只有 `opacity_useFreePathModel = .true.` 时，FLASH 才尝试模型。
2. helper 还会检查网页 `/api/flash/control`，只有网页端 `enabled=true` 且当前 Z 在 `modelElements` 中才使用模型。
3. 只对单元素 species 尝试模型；DD 等复合或未训练元素会走原 IONMIX 表。
4. 模型返回的是自由程 `lnu`，helper 转成 `opacityRO = 1 / (rho * lnu)`，其中 `rho` 是 species 质量密度。
5. helper、网页、元素开关、模型超出训练范围或模型任一失败时，`op_getSpeciesROTableOpacity` 自动落回原来的表插值。
6. FLASH 侧已有 4096 项轻量缓存，按 `(Z, log10(rho), log10(T))` 的 0.001 精度复用结果。命中模型的点会缓存 `opacityRO`；超范围、网页关闭、元素不允许或 helper 失败的点也会缓存“回退查表”状态，避免同一个无效点反复启动 Python/helper。
7. 目前 `tgama` 用 `speciesTemperature` 做等温近似，还没有把 `TRAD_VAR` 从 `Opacity(soln,...)` 单独传到 Rosseland 表函数。

已验证：

```text
cd /home/user2/Haigerloch/FLASH4.8/improved_MRT
make -j2
SUCCESS
```

helper 验证结果：

```text
网页 FLASH 控制关闭时：RuntimeError: FLASH free-path model control is disabled
网页 FLASH 控制开启且 Z=4、rho=1、Te=Trad=100 eV 时：kappa_R = 4.71731585983843388e+02 cm^2/g
模型范围外时：RuntimeError: free-path model prediction is outside the training range
```

网页模式验证脚本：

```bash
cd /home/user/Rosseland_opa_new_hetero/equally
python3 verify_flash_modes.py
```

当前已验证：

```text
model        -> 使用模型
hybrid 命中   -> 使用表中真实 lnu
truth 命中    -> 使用表中真实 lnu
hybrid 未命中 -> 使用模型兜底
truth 未命中  -> 正常报错
```

## 使用方式

网页服务先保持运行：

```bash
cd /home/user/Rosseland_opa_new_hetero/equally
/home/user/miniconda3/envs/ai/bin/python free_path_web_app.py \
  --host 0.0.0.0 \
  --port 8790 \
  --model /home/user/Rosseland_opa_new_hetero/equally/free_path_model_outputs/unified_free_path_model.npz
```

当前如果已经启动，可用 `cat free_path_web_app.pid` 和 `curl http://127.0.0.1:8790/health` 检查。

在网页 `http://127.0.0.1:8790` 的“不透明度自由程”页里使用“FLASH 接入控制”栏，打开开关、选择策略、勾选允许模型兜底的元素。当前建议先只勾选 `Z=4 Be` 做小范围测试。

然后切到“FLASH 控制台”页签：

1. 选择运行目录，例如 `improved_MRT`。
2. 如果要做新算例，先在“新建项目”里填写项目名，选择模板项目和复制模式。
3. 在“全部 flash.par 参数”里搜索并修改任意参数；布尔值用开关，数字/字符串用输入框，保存时只写已修改项并自动备份旧 `flash.par`。
4. 可以直接点“启动 FLASH”，也可以在“命令行”里输入原本会在终端运行的命令，例如 `./flash4`、`mpirun -np 2 ./flash4`、`tail -n 80 ZPinch_2D.log`。
5. 运行过程中用“刷新”查看状态，用“日志和结果”直接打开 `ZPinch_2D.log`、文本/表格/图片文件。
6. 对 FLASH HDF5 plot/checkpoint 结果，选择文件、变量、线性或 `log10` 尺度，然后点“显示图像”，网页会直接显示二维结果图。
7. 对 `trajectory.dat`、`circuit.dat`、`current.dat`、CSV/TXT 等数字表，选择 X 列和 Y 列后点“显示图像”，网页会直接显示曲线图。

在 `/home/user2/Haigerloch/FLASH4.8/improved_MRT/flash.par` 中至少确认：

```text
useRadTrans = .true.
useOpacity = .true.
opacity_useFreePathModel = .true.
```

然后按原来的方式运行 FLASH，例如：

```bash
cd /home/user2/Haigerloch/FLASH4.8/improved_MRT
./flash4
```

或按需要用 MPI：

```bash
mpirun -np 2 ./flash4
```

关闭模型时，把网页 FLASH 接入控制关掉，或把 `flash.par` 中 `opacity_useFreePathModel = .false.`。两边任一关闭都会回到原 IONMIX/查表。当前网页 FLASH 控制已经恢复为关闭状态。

## 后续性能优化

- 不建议每个 cell 每个能群直接 HTTP 调 `/api/predict`。FLASH opacity 调用频率很高，首次访问的新状态点仍会比原生查表慢。
- 当前 Fortran 侧已有 4096 项正/负缓存：重复命中的模型点不再调用 helper；重复的超范围或关闭状态也不再调用 helper，直接走原表插值。热缓存路径已经接近查表加一次缓存键检查。
- 若要让任意首次模型点也达到或超过查表速度，需要把模型编译进 FLASH 进程，或预先生成本地 opacity/free-path 表让 FLASH 直接查表；单纯通过网页/Python helper 无法保证每个新点都快过原生表插值。
- 正式长算例推荐做一个常驻 sidecar、C/Fortran 可调用模型，或直接生成 IONMIX/本地表，并支持真正的批量预测。
- 因网页模型没有能群维度，若用于 MGD，每个能群会拿到同一个 Rosseland mean 转换值。这是物理近似，需要确认。

## 探针脚本

已添加：

```text
/home/user/Rosseland_opa_new_hetero/equally/flash_free_path_opacity_probe.py
```

示例，把 FLASH 初始 Be liner 条件转换成网页模型输入并输出 `kappa_R`：

```bash
cd /home/user/Rosseland_opa_new_hetero/equally
python3 flash_free_path_opacity_probe.py \
  --Z 4 \
  --rho 1.848 \
  --tele-k 5802.26105 \
  --trad-k 5802.26105
```

示例，直接用模型标准坐标：

```bash
python3 flash_free_path_opacity_probe.py \
  --model-coordinates \
  --Z 13 --rod 0 --tep 3 --tgama 3
```

FLASH 当前直接版已经会调用这个 helper，并在 Fortran 侧做轻量缓存。它适合先跑小规模连通性和物理敏感性测试；正式长算例应改成批量/常驻模型服务。
