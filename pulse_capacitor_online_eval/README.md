# 脉冲电容器状态在线评估服务

这是根据 `202610366208.8-脉冲电容器状态在线评估方法及装置-申请文件.docx` 实现的网页端服务。后端使用标准库 Python，前端使用原生 HTML/CSS/JS，不依赖外部 Web 框架。

## 已实现功能

- 读取真实公开 CSV 数据集，并在网页端选择电容样本和参数列。
- 对每个参数序列执行级比检验。
- 建立 GM(1,1) 灰色预测模型，计算发展系数 `a` 和灰作用量 `b`。
- 根据电容量下降阈值、ESR 增长阈值预测临界步数和寿命。
- 计算平均相对精度。
- 输出单参数结果与多参数综合结果。
- 按“已运行时间/预测寿命 >= 阈值”给出预警状态。
- 对历史数据做滚动验证：每次取连续窗口建模，向后预测指定步数，再和真实终点比较。
- 绘制实测与预测趋势图，支持导出 JSON 报告。
- 在左侧“全部模型对比预测”页中统一比较专利 GM(1,1)、统计模型、基线、量化感知混合模型以及 GRU、LSTM、TCN、Transformer；每个模型只用验证集选择最佳建模步数，再在相同独立测试集按 NMAE 排名，并可逐模型、逐严格特征查看实测/预测曲线和寿命预测。
- 在右侧“专利 GM(1,1)”页中保留申请文件算法的单独评估、级比检验、滚动验证与工程预警结果。

## 数据来源

当前内置数据位于 `data/raw/`，共 21 个 CSV 文件，来自：

- Data.gov: <https://catalog.data.gov/dataset/high-temperature-evaluation-of-tantalum-capacitors-test-1-3865b>
- OEDI landing page: <https://data.openei.org/submissions/6751>
- 单个 CSV 示例: <https://gdr.openei.org/files/440/cap2-6.csv>

这些数据是 Sandia National Laboratories 发布的钽电容高温老化测试数据，包含 `Time`、不同频率下的 `C` 和 `ESR` 测量值。它是真实电容老化数据，但不是脉冲电容器逐次放电波形数据。

用户提供的 `ceshishuju.zip` 是 OWON 示波器逐次脉冲波形。严格按申请文件使用时，先将每次波形提取为运行参数序列：

```bash
cd /home/user/pulse_capacitor_online_eval
python3 tools/convert_owon_spbxds_to_patent_csv.py \
  ceshishuju.zip \
  data/raw/ceshishuju_patent_features.csv
```

生成的 `ceshishuju_patent_features.csv` 包含：

- `VoltageMaxRaw`: 每次脉冲 CH1 电流换算波形的最大幅值。
- `VoltageFirstZeroTimeUs`: 从脉冲起点到主峰后首次回到零轴的时间。
- `VoltageReversePeakCoefficient`: CH1 反向幅值 / 最大幅值。
- `VoltageMinAbsRaw`: 主峰后的最小值绝对幅值。
- `DischargePeriodSec`: 相邻两次脉冲采集时间间隔。

已确认实验采集时 CH1 启用了电流换算，因此这些特征按电流换算波形解释，不是电容器端真实电压。`Voltage*Raw` 是为兼容既有 CSV/API 保留的历史字段名；固定比例单位换算不改变 NMAE 和模型排名，本版继续保留原始采样量纲。

## 启动方法

```bash
cd /home/user/pulse_capacitor_online_eval
python3 app.py --host 127.0.0.1 --port 8890
```

浏览器打开：

```text
http://127.0.0.1:8890
```

## 接口方法

列出数据集：

```bash
curl http://127.0.0.1:8890/api/datasets
```

运行评估：

```bash
curl -X POST http://127.0.0.1:8890/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "cap1-1.csv",
    "columns": ["C1kHz", "ESR1kHz"],
    "window": 80,
    "horizon": 1200,
    "parameterRules": {
      "C1kHz": {"direction": "decrease", "percent": 5},
      "ESR1kHz": {"direction": "increase", "percent": 100}
    },
    "accuracyThreshold": 90,
    "lifeRatioThreshold": 90,
    "smoothing": false
  }'
```

## 参数说明

- `dataset`: 数据文件名。
- `columns`: 参与评估的参数列，默认建议选择 `C1kHz` 和 `ESR1kHz`。
- `window`: 使用最近多少个点建模，最少 4。
- `horizon`: 按“预测方式 = 按步数”时使用的向后预测步数。
- `predictionTime`: 按“预测方式 = 按时间”时使用的状态预测时长，单位与 CSV 的 `Time` 列一致，程序会自动换算为步数。
- `parameterRules`: 每个参数自己的失效规则。`direction` 可为 `decrease` 或 `increase`，`percent` 表示相对原始数据第 1 次放电值变化多少百分比达到临界。放电周期在第 1 次放电时尚不存在，因此使用第 1 次至第 2 次之间的首个有效间隔。
- `accuracyThreshold`: 平均相对精度阈值，默认 90%。
- `lifeRatioThreshold`: 寿命比阈值，默认 90%。
- `validationHorizon`: 按“验证方式 = 按步数”时使用的历史验证向后预测步数，默认 20。
- `validationTime`: 按“验证方式 = 按时间”时使用的历史验证向后预测时长，单位与 CSV 的 `Time` 列一致，程序会自动换算为步数。
- `validationErrorThreshold`: 历史验证允许 MAPE，默认 5%。
- `validationStartIndex`: 回测起始训练行，默认 1。
- `validationEndIndex`: 回测结束真实对比行；为空时自动用数据末行。
- `validationStride`: 滑动间隔，默认 1，表示每次窗口向后移动 1 行。
- `smoothing`: 是否启用移动加权平滑。

## 不确定项

- 寿命比采用工程预警逻辑：`已运行时间 / 预测寿命 >= 阈值` 触发预警。
- 申请文件同时出现“算术平均”和“加权平均”融合描述。当前实现采用等权平均。
- 公开数据没有逐次脉冲放电的最大电压、最小电压、最大电流、最小电流、反峰系数和首次过零时间。当前以内置真实 C/ESR 老化数据演示模型流程，接入现场系统时应替换或扩展 CSV 字段。
- Word 公式抽取后部分上下标缺失，GM(1,1) 公式按正文和灰色模型标准形式实现。

## 历史滚动验证

验证逻辑：

```text
第 1 至 80 条数据建模 -> 向后预测 N 步 -> 与第 80+N 条真实数据比较
第 2 至 81 条数据建模 -> 向后预测 N 步 -> 与第 81+N 条真实数据比较
...
```

其中回测起始行、回测结束行、滑动间隔都可以在网页左侧调节。比如起始行 `5`、结束行 `140`、滑动间隔 `2` 表示从第 5 行开始建模，每次窗口向后移动 2 行，真实对比目标不超过第 140 行。

左侧面板可以分别把预测方式和验证方式切换为“按步数”或“按时间”。按时间时，预测时长和验证时长会按数据集 `Time` 列的中位采样间隔换算成步数。例如 `Time` 相邻间隔约为 `1.23`，填写验证时长 `12.5`，程序会换算成向后预测 `ceil(12.5 / 1.23) = 11` 步。

输出指标：

- `MAE`: 平均绝对误差。
- `RMSE`: 均方根误差。
- `MAPE`: 平均绝对百分比误差。
- `最大误差`: 所有滚动验证中最大的绝对百分比误差。
- `最后预测值 / 最后真实值`: 最后一轮滚动窗口的预测终点和真实终点。

图表坐标轴：

- 状态预测图：横轴是样本序号，纵轴是参数值。
- 历史验证图：横轴是真实对比行号，纵轴是绝对百分比误差。
- 状态预测图中实线表示真实/实测数据，虚线表示模型预测数据。
- 历史验证图中实线表示每一轮回测的预测误差。

## 全部模型对比预测

结果区顶部有两个主栏目：

- `全部模型对比预测`：合并原多模型与严格特征深度学习页面，在同一独立测试口径下比较全部方法，位于左侧。
- `专利 GM(1,1)`：保留申请文件的灰色模型、级比、`a`、`b`、寿命与预警结果。

统一候选方法共 11 个：

- `GM(1,1) 灰色模型`：申请文件方法，作为基准并参与最终选择。
- `持续值预测`：未来保持为窗口最后一个实测值。
- `线性趋势回归`：最小二乘直线外推。
- `阻尼 Holt 趋势`：估计水平与趋势，并让长期趋势逐渐衰减；平滑参数只使用当前训练窗口选择。
- `AR(1) 自回归`：根据相邻记录的一阶相关关系递推未来值。
- `Ridge 线性基线`：用五个严格特征的历史窗口联合预测。
- `量化感知混合模型`：按主幅值上包络、过零时间中位数和反向幅值众数处理离散档位，并由两种幅值重建反向峰系数。
- `GRU、LSTM、TCN、Transformer`：五特征联合输入和联合输出的深度模型。

模型选择不是按建模窗口内拟合精度进行，而是按历史滚动回测 MAPE 进行。选择后再使用最近一个建模窗口重建该方法，生成当前未来预测曲线。对于接近 0 的参数，相对误差会因分母很小而显著放大，应同时查看 MAE、RMSE 和原始数据量级。

比较接口：

```text
POST /api/compare-models
```

请求参数与 `/api/evaluate`、`/api/validate` 相同。

## 严格特征统一模型测试

统一模型测试只使用 `ceshishuju_patent_features.csv`，不会使用公开 C/ESR 数据。五个特征来自 CH1 电流换算波形：

- CH1 最大幅值。
- CH1 首次过零时间。
- CH1 反向峰系数。
- CH1 反向最小幅值绝对值。
- 放电周期。

固定预测目标为 20 步后；第 1-424 行用于训练，第 425-515 行用于选择模型自己的建模窗口，第 516-606 行作为独立测试集。每个模型从 `8、16、24、32、48、64、80、120` 步中按验证 NMAE 选择自己的最佳窗口，测试集不参与训练、早停或窗口选择。四个深度架构各使用三个随机种子集成。

网页默认显示全部 11 个模型中独立测试 NMAE 最低者的预测曲线。可以任意选择模型和五个特征；每个组合都在同一图表中用实线显示目标行真实值、虚线显示模型预测值。模型表按测试 NMAE 从低到高排列，并显示每个模型由验证集选出的最佳建模步数。图表悬停时显示横纵辅助线、目标行、真实值和预测值。

当前搜索结果：量化感知混合模型由验证集选中 32 步，独立测试 NMAE 为 `10.68%`，归一化精度为 `89.32%`，在全部方法中排名第 1；GRU 测试精度为 `84.44%`，专利 GM(1,1) 的最佳窗口为 80 步、测试 NMAE 为 `19.72%`。

寿命阈值固定以原始数据第 1 次放电为基准，当前状态采用部署模型稳健估计。放电周期使用第 1 次至第 2 次之间的首个有效间隔。多参数健康度按独立测试误差和量化分辨率加权；有限寿命还必须同时满足参数可靠度、至少两个窗口同向退化以及在经验证时距内达到阈值，系统寿命取可信参数中最早失效者。

参考电容在 606 次记录后继续 1500 次失效，总寿命 2106 次。Theil-Sen 趋势反标得到：`VoltageMaxRaw` 上升 `6.98%`，`VoltageFirstZeroTimeUs` 下降 `23.32%`。该标签不计入独立测试精度。

程序对 1-500 步执行独立滚动回测，综合精度不低于 `85%` 的最长时距为 `120` 步（精度 `87.46%`）。因此当前只可报告剩余寿命 `>120 步`、总寿命 `>726 步`；5000 步曲线是探索性外推。图中绿色为实测、蓝色为经验证预测、棕色为探索外推、红色为阈值。

普通 MAPE 在真实值接近 0 时会失真，因此模型选择使用五个参数平均 NMAE：

```text
单参数 NMAE = MAE / 该参数训练集范围 × 100%
平均 NMAE = 五个参数 NMAE 的算术平均
归一化精度 = 100% - 平均 NMAE
```

重新训练：

```bash
cd /home/user/pulse_capacitor_online_eval
/home/user/miniconda3/envs/ai/bin/python tools/train_strict_deep_models.py
```

训练报告生成到 `data/models/strict_deep_report.json`，网页通过以下接口读取：

```text
GET /api/deep-learning-report
```

## 测试

首次运行浏览器测试时安装依赖和 Chromium：

```bash
cd /home/user/pulse_capacitor_online_eval
npm install
npx playwright install chromium
```

一键运行后端和浏览器全部回归：

```bash
npm run test:all
```

也可以分别运行：

```bash
npm run test:backend
npm run test:ui
```

`test:ui` 会自动在 `127.0.0.1:8891` 启动临时服务，逐一检查 11 个模型 × 5 个参数的真实/预测曲线、悬停十字线、两个页签及移动端布局。若要检查已经运行的服务，可执行 `BASE_URL=http://127.0.0.1:8890 npm run test:ui`。

详细测试记录见 `docs/TEST_REPORT.md`。
