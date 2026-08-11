# 算法实现说明

## 输入序列

对每个运行参数单独建模：

```text
x0 = (x0(1), x0(2), ..., x0(n))
```

当前服务要求序列为正数，原因是 GM(1,1) 的级比检验和指数响应形式要求正值序列。

## 波形特征提取

申请文件中的“通过波形预测寿命”不是直接对整条波形建模，而是先对每次脉冲放电波形提取运行参数，再对每个运行参数形成序列。

对 `ceshishuju.zip` 这类 OWON SPBXDS 波形文件，转换脚本为：

```bash
python3 tools/convert_owon_spbxds_to_patent_csv.py \
  ceshishuju.zip \
  data/raw/ceshishuju_patent_features.csv
```

每个 bin 文件对应一次脉冲，脚本提取：

```text
VoltageMaxRaw                  CH1 电流换算波形最大幅值的原始采样量
VoltageMinAbsRaw               主峰后反向最小幅值的绝对值
VoltageReversePeakCoefficient  CH1 反向幅值 / 最大幅值
VoltageFirstZeroTimeUs         CH1 主峰后首次回到零轴的时间
DischargePeriodSec             相邻两次脉冲采集时间间隔
```

已确认当前实验采集时 CH1 启用了电流换算，因此这些列按电流换算波形解释，不是电容器端真实电压。`Voltage*Raw` 是兼容既有 CSV/API 的历史字段名；数值仍保留原始采样量纲。

## 量化感知预测

当前 606 条记录中的主幅值只有 4 个采样档位，反向最小幅值只有 2 个档位。量化感知混合模型不把这些量强行视为平滑连续值，而是使用：

```text
最大幅值预测       = 最近窗口的上包络
首次过零时间预测   = 最近窗口的中位数
反向最小幅值预测   = 最近窗口的众数档位
反向峰系数预测     = 预测反向幅值 / 预测最大幅值
放电周期预测       = 训练段有效周期中位数
```

窗口只在验证集上从 `8、16、24、32、48、64、80、120` 中选择。验证指标完全并列时取并列窗口的中间值，因此当前部署窗口为 32 步。独立测试集只用于最终审计，不参与模型或寿命方法选择。

## 多参数剩余寿命

全部模型页以原始数据第 1 次放电值 `B_i` 作为固定阈值基准，部署模型给出的稳健当前值为 `C_i`。放电周期在第 1 次时不可计算，使用第 1 次至第 2 次之间的首个有效间隔。按参数独立设置的方向和变化率计算阈值 `M_i`，不再用当前值或建模窗口移动阈值。

```text
下降失效: M_i = B_i * (1 - p_i)
上升失效: M_i = B_i * (1 + p_i)
状态消耗 d_i = clip(朝失效方向的 (C_i-B_i) / abs(M_i-B_i), 0, 1)
参数权重 w_i = 独立测试精度 * 量化分辨率系数
多参数健康度 = (1 - sum(w_i*d_i) / sum(w_i)) * 100%
```

有限剩余寿命只在以下条件同时成立时输出：参数可靠度至少 35%，32/64/120/200 步窗口中至少两个窗口显示同向退化且包含一个不大于 120 步的近期窗口，模型在经回测验证的时距内达到阈值。多个参数满足时取最早阈值交叉点，因为任一关键参数失效即可限制系统寿命。

独立测试对 `1、5、10、20、40、80、120、160、200、240、300、400、500` 步分别回测。综合归一化精度至少 85% 的最长时距为 120 步，精度 87.46%；160 步降为 83.99%。所以当前结论是剩余寿命 `>120` 步、总寿命 `>726` 步。5000 步没有交叉仅是探索性外推，不是已验证寿命。

## 级比检验

```text
sigma(k) = x0(k-1) / x0(k), k = 2, ..., n
cover = (exp(-2/(n+1)), exp(2/(n+1)))
```

所有级比都落入覆盖区间时认为序列通过有效性检验。

## GM(1,1)

累加生成：

```text
x1(k) = sum(x0(i)), i = 1, ..., k
```

背景值：

```text
z1(k) = 0.5 * (x1(k) + x1(k-1))
```

模型：

```text
x0(k) + a * z1(k) = b
```

矩阵形式：

```text
Y = [x0(2), x0(3), ..., x0(n)]^T
B = [[-z1(2), 1], [-z1(3), 1], ..., [-z1(n), 1]]
[a, b]^T = (B^T B)^-1 B^T Y
```

时间响应：

```text
x1_hat(k) = (x0(1) - b/a) * exp(-a * (k - 1)) + b/a
x0_hat(k) = x1_hat(k) - x1_hat(k - 1), k >= 2
```

## 精度

```text
epsilon(k) = x0(k) - x0_hat(k)
delta(k) = abs(epsilon(k)) / abs(x0(k))
accuracy = (1 - mean(delta)) * 100%
```

## 临界寿命

服务优先用预测序列搜索临界交叉点，避免闭式公式在 `a` 接近 0 或趋势反向时产生不可用结果。

每个参数都有自己的失效规则：

```json
{
  "parameterRules": {
    "VoltageMaxRaw": {"direction": "increase", "percent": 6.98},
    "VoltageFirstZeroTimeUs": {"direction": "decrease", "percent": 23.32}
  }
}
```

实验先验已确认：CH1 最大幅值随老化上升，首次过零时间随老化下降。参考电容在当前 606 次记录后又运行 1500 次失效，总寿命为 2106 次。对 606 次观测使用 Theil-Sen 稳健趋势并外推到第 2106 次，再相对第 1 次放电基准反标，得到最大幅值上升 `6.98%`、首次过零时间下降 `23.32%`。这是使用已知失效标签进行的阈值标定，不能作为独立预测准确率；两个阈值仍可在界面单独调整。

```text
稳健斜率 s = median((x_j-x_i)/(j-i)), i < j
失效点估计 M = median(x_k-s*k) + s*2106
上升比例 = (M/B-1)*100%
下降比例 = (1-M/B)*100%
```

临界值计算：

```text
direction = decrease: M = x0(1) * (1 - percent / 100)
direction = increase: M = x0(1) * (1 + percent / 100)
```

如果接口没有传 `parameterRules`，服务仍保留旧的电容量下降、ESR 增长和通用增长阈值作为兼容默认值。

预测寿命：

```text
T = k_critical * period
```

其中 `period` 默认使用数据集 `Time` 列的中位采样间隔；没有 `Time` 列时取 1。

## 预警判断

当前实现采用工程逻辑：

```text
accuracy < accuracy_threshold
or elapsed_life / predicted_life >= life_ratio_threshold
=> warning
```

也就是模型精度低于阈值，或者已运行寿命占预测寿命的比例达到阈值时预警。

## 历史滚动验证

历史验证用于回答“模型拿前面的数据预测未来，和真实未来比准不准”。

设建模窗口为 `w`，验证步数为 `h`，滑动间隔为 `s`，训练窗口起点为 `i`：

如果用户填写的是预测时长 `T`，程序先按 `Time` 列的中位采样间隔 `period` 换算：

```text
h = ceil(T / period)
```

```text
window_i = x(i), x(i+1), ..., x(i+w-1)
target_i = x(i+w+h-1)
prediction_i = GM11(window_i).forecast(w+h)
error_i = prediction_i - target_i
abs_percent_error_i = abs(error_i) / abs(target_i) * 100%
i_next = i + s
```

程序在指定的回测起始行和结束行之间滚动计算，并输出：

```text
MAE = mean(abs(error))
RMSE = sqrt(mean(error^2))
MAPE = mean(abs_percent_error)
```

历史验证图的横轴是真实对比行号 `i+w+h-1`，纵轴是绝对百分比误差。
