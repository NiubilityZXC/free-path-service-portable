# 真实数据来源

## 采用的数据

项目内置的 21 个 CSV 文件来自 Sandia National Laboratories 发布的公开数据集：

```text
High Temperature Evaluation of Tantalum Capacitors - Test 1
```

公开页面：

- Data.gov: <https://catalog.data.gov/dataset/high-temperature-evaluation-of-tantalum-capacitors-test-1-3865b>
- OEDI: <https://data.openei.org/submissions/6751>
- CSV 示例: <https://gdr.openei.org/files/440/cap2-6.csv>

Data.gov 页面说明该数据集用于评价 260C 下三类钽电容，CSV 资源描述为 260C 和 5V 偏置下的电容量与 ESR 测量值。

## 为什么选择它

申请文件的目标是脉冲电容器在线状态评估，核心建模对象是能反映电容退化趋势的运行参数。公开可直接下载的“逐次脉冲放电波形”数据很少；该数据集虽然不是脉冲放电波形，但包含真实电容老化过程中的电容量和 ESR 序列，可用于验证 GM(1,1) 建模、趋势预测、临界阈值和预警流程。

## 备选但未采用的数据

NASA PCoE 的 `Capacitor Electrical Stress` 数据集包含 EIS 和充放电信号数据，页面说明电容在 10V、12V、14V 三个电应力水平下测试。公开页面：

<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

该压缩包约 4.8GB，本次交付未全量下载，避免把项目变成大型数据归档。后续若需要脉冲/充放电波形级验证，可以单独下载并编写抽取器。
