const state = {
  datasets: [],
  latestEvaluation: null,
  latestValidation: null,
  latestModelComparison: null,
  latestDeepLearning: null,
  activePage: "deepLearningPage",
};
const STRICT_DATASET = "ceshishuju_patent_features.csv";
const STRICT_CALIBRATED_RULES = {
  VoltageMaxRaw: { direction: "increase", percent: 6.98 },
  VoltageFirstZeroTimeUs: { direction: "decrease", percent: 23.32 },
};
const palette = ["#0e766d", "#2764a8", "#b45f06", "#6f4fa1", "#aa2b2b", "#3f7d20"];
const $ = (id) => document.getElementById(id);
const PARAM_META = {
  C100Hz: {
    label: "100Hz 电容量",
    desc: "100Hz 测试频率下的电容量，下降通常表示容量衰减。",
  },
  C1kHz: {
    label: "1kHz 电容量",
    desc: "1kHz 测试频率下的电容量，下降通常表示容量衰减。",
  },
  C10kHz: {
    label: "10kHz 电容量",
    desc: "10kHz 测试频率下的电容量，下降通常表示容量衰减。",
  },
  ESR100Hz: {
    label: "100Hz 等效串联电阻",
    desc: "100Hz 测试频率下的 ESR，增长通常表示损耗或内阻上升。",
  },
  ESR1kHz: {
    label: "1kHz 等效串联电阻",
    desc: "1kHz 测试频率下的 ESR，增长通常表示损耗或内阻上升。",
  },
  ESR10kHz: {
    label: "10kHz 等效串联电阻",
    desc: "10kHz 测试频率下的 ESR，增长通常表示损耗或内阻上升。",
  },
  VoltageMaxRaw: {
    label: "CH1 最大幅值（电流换算采样）",
    desc: "每次脉冲 CH1 波形相对基线的最大幅值。按上升失效判断；根据现有数据后再运行 1500 次失效的结果，默认反标为上升 6.98%。",
  },
  VoltageFirstZeroTimeUs: {
    label: "CH1 首次过零时间（微秒）",
    desc: "从脉冲起点到主峰后首次回到零轴的时间。按下降失效判断；根据实测失效次数，默认反标为下降 23.32%。",
  },
  VoltageReversePeakCoefficient: {
    label: "CH1 反向峰系数",
    desc: "主峰后反向幅值与最大幅值的比值；量化感知模型由两种预测幅值重算该系数。",
  },
  VoltageMinAbsRaw: {
    label: "CH1 反向最小幅值绝对值（电流换算采样）",
    desc: "主峰后反向最小值的绝对幅值。该批数据主要表现为两个 ADC 量化档位。",
  },
  DischargePeriodSec: {
    label: "放电周期（秒）",
    desc: "相邻两次脉冲采集时间间隔。第 1 次放电尚无间隔，因此阈值基准取第 1 次至第 2 次之间的首个有效间隔。",
  },
  PeakADC: {
    label: "波形峰值（ADC）",
    desc: "每次波形的最大 ADC 采样值，属于通用波形统计特征。",
  },
  PeakToPeakADC: {
    label: "峰峰值（ADC）",
    desc: "每次波形最大值与最小值的差，属于通用波形统计特征。",
  },
  RMSADC: {
    label: "均方根值（ADC）",
    desc: "每次波形的 RMS 值，用于表示有效幅值大小。",
  },
  MeanAbsADC: {
    label: "平均绝对值（ADC）",
    desc: "每次波形绝对值的平均值，用于表示整体幅值水平。",
  },
  EnergyADC2: {
    label: "平均能量特征（ADC²）",
    desc: "每次波形采样值平方的平均值，用于描述波形能量趋势。",
  },
  PulseWidthSamples: {
    label: "脉宽（采样点数）",
    desc: "高于阈值的采样点数量，用于描述脉冲宽度变化。",
  },
  TroughAbsADC: {
    label: "谷值绝对值（ADC）",
    desc: "每次波形最低点的绝对值，属于通用波形统计特征。",
  },
};
const INPUT_HELP = [
  ["数据集", "选择要分析的 CSV 数据。严格波形数据请选择 ceshishuju_patent_features.csv。"],
  ["参数列", "选择参与预测、回测和寿命判断的运行参数。全部模型页使用固定五个严格特征统一比较，专利页使用当前勾选参数。"],
  ["建模窗口", "主预测使用最近多少行数据建模。例如 80 表示取最后 80 次记录作为当前模型输入。"],
  ["预测方式", "按步数表示直接填写未来步数；按时间表示填写时长，程序按 Time 列采样间隔换算成步数。"],
  ["预测步数", "主预测向未来外推多少步，只在预测方式为按步数时生效。"],
  ["预测时长", "主预测向未来外推多少时间，只在预测方式为按时间时生效，单位与 Time 列一致。"],
  ["单参数失效阈值", "每个参数都有独立方向和百分比；临界值固定按原始数据第 1 次放电计算。"],
  ["精度阈值 %", "模型对建模窗口内已知数据的拟合精度低于该值时给出预警。"],
  ["寿命比阈值 %", "已运行时间 / 预测总寿命达到该比例时给出预警。"],
  ["验证方式", "历史滚动验证按步数或按时间决定每轮往后预测多远。"],
  ["验证步数", "每轮回测从训练窗口末尾向后预测多少步，并与对应真实行比较。"],
  ["验证时长", "每轮回测向后预测多少时间，程序按 Time 列采样间隔换算成步数。"],
  ["允许误差 %", "历史滚动验证的平均 MAPE 小于等于该值时判定通过。"],
  ["回测起始行", "历史滚动验证从哪一行开始取训练窗口。"],
  ["回测结束行", "历史滚动验证真实对比目标最多到哪一行；为空时自动用数据末行。"],
  ["滑动间隔", "每轮回测训练窗口向后移动的行数。1 表示逐行滚动。"],
  ["启用移动加权平滑", "对输入序列先做平滑再建模，用于降低噪声，但会弱化突变。"],
  ["寿命搜索上限", "深度学习寿命预测最多向未来检查多少步，当前训练报告最多支持 5000 步。"],
];
const OUTPUT_HELP = [
  ["工程状态", "精度和寿命比共同判断的最终状态，正常或预警。"],
  ["寿命比", "已运行时间 / 预测总寿命 × 100%。越高表示越接近预测临界点。"],
  ["精度", "所有选中参数的单参数拟合精度平均值。"],
  ["预测寿命", "预测从建模窗口起点到首次达到临界值的总时间；未碰到临界值时显示未到临界。"],
  ["参数", "当前参与建模的运行参数，界面显示中文名并保留 CSV 字段名。"],
  ["方向", "该参数按上升达到临界还是下降达到临界。"],
  ["第 1 次放电基准", "直接取原始 CSV 第 1 行；放电周期因第 1 次尚无间隔，取第 2 行首个有效间隔。"],
  ["临界值", "由原始数据第 1 次放电值和阈值比例计算；放电周期使用首个有效间隔。"],
  ["a", "GM(1,1) 发展系数，决定趋势方向和变化快慢。"],
  ["b", "GM(1,1) 灰作用量，可近似理解为模型常数项。"],
  ["精度", "单个参数在当前建模窗口内拟合真实值的平均相对精度。"],
  ["级比", "GM(1,1) 输入序列有效性检验结果；异常说明序列波动较强。"],
  ["剩余步数", "从建模窗口最后一行开始，到预测曲线首次达到临界值还剩多少步；负数表示窗口内已越过。"],
  ["平均 MAPE", "历史滚动验证中所有选中参数的平均绝对百分比误差。"],
  ["验证窗口", "每轮回测使用多少行历史数据建模。"],
  ["向后步数", "每轮回测向后预测多少步后与真实值比较。"],
  ["回测范围", "用于历史滚动验证的真实数据行号范围。"],
  ["验证次数", "某个参数实际完成了多少轮滚动验证。"],
  ["MAE", "历史验证平均绝对误差，保留原参数量纲。"],
  ["RMSE", "历史验证均方根误差，对大误差更敏感。"],
  ["MAPE", "历史验证平均绝对百分比误差，越小表示外推越准。"],
  ["最大误差", "单轮最大绝对百分比误差。"],
  ["最后目标行", "末轮回测目标行号。"],
  ["最后预测值", "末轮预测值。"],
  ["最后真实值", "末轮真实值。"],
  ["最佳回测精度", "100% 减去各参数最佳模型的平均滚动回测 MAPE。"],
  ["最优方法", "当前参数在相同滚动回测条件下 MAPE 最低的候选方法。"],
  ["降低百分点", "GM 回测 MAPE 减去最佳方法 MAPE；正数表示新选择的方法误差更低。"],
  ["剩余寿命结论", "有限值或经验证下限。"],
  ["经验证总寿命下限", "已运行步数加剩余下限。"],
  ["限制寿命参数", "最早到阈值的可信参数。"],
  ["参考样本实测失效", "该电容在现有 606 次记录后又运行 1500 次失效，总寿命为 2106 次；该结果用于反标阈值，不作为独立预测精度。"],
];
function optionalNumber(id) {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
}
function paramMeta(column) {
  return PARAM_META[column] || { label: column, desc: "CSV 中的数值参数列。" };
}
function paramLabel(column) {
  return paramMeta(column).label;
}
function paramFullLabel(column) {
  const meta = paramMeta(column);
  return meta.label === column ? column : `${meta.label}（${column}）`;
}
function defaultRuleForColumn(column) {
  const lower = column.toLowerCase();
  if (STRICT_CALIBRATED_RULES[column]) return { ...STRICT_CALIBRATED_RULES[column] };
  if (lower.startsWith("c") && !lower.includes("esr")) return { direction: "decrease", percent: 5 };
  if (lower.includes("esr")) return { direction: "increase", percent: 100 };
  if (/voltagemax|currentmax/.test(lower)) return { direction: "increase", percent: 5 };
  if (lower.includes("firstzerotime")) return { direction: "decrease", percent: 20 };
  if (/voltageminabs|currentminabs|reversepeak/.test(lower)) return { direction: "decrease", percent: 5 };
  return { direction: "increase", percent: 20 };
}
function datasetLabel(item) {
  if (item.name === "ceshishuju_patent_features.csv") {
    return "实验波形严格特征";
  }
  if (item.name === "ceshishuju_waveform_features.csv") {
    return "实验波形统计特征";
  }
  return item.name;
}
function sourceText(dataset) {
  if (!dataset) return "真实数据：待选择";
  if (dataset.name === "ceshishuju_patent_features.csv") {
    return "真实数据：OWON 示波器 CH1 电流换算波形，已提取申请文件所列运行参数";
  }
  if (dataset.name === "ceshishuju_waveform_features.csv") {
    return "真实数据：OWON 示波器脉冲波形统计特征";
  }
  return "真实数据：Sandia / OEDI 钽电容高温老化 CSV";
}
function selectedMode(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "steps";
}
function syncModeControls() {
  const predictionByTime = selectedMode("predictionMode") === "time";
  $("horizon").disabled = predictionByTime;
  $("predictionTime").disabled = !predictionByTime;
  const validationByTime = selectedMode("validationMode") === "time";
  $("validationHorizon").disabled = validationByTime;
  $("validationTime").disabled = !validationByTime;
}
function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const number = Number(value);
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.001) return number.toExponential(2);
  return number.toFixed(digits);
}
function fmtExact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const number = Number(value);
  const magnitude = Math.abs(number);
  const digits = magnitude >= 1000 ? 4 : magnitude >= 0.1 ? 6 : magnitude >= 0.0001 ? 9 : 12;
  return Number(number.toFixed(digits)).toLocaleString("zh-CN", {
    useGrouping: false, maximumFractionDigits: digits,
  });
}
function fmtThreshold(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const number = Number(value);
  const magnitude = Math.abs(number);
  const digits = magnitude >= 1000 ? 4 : magnitude >= 0.1 ? 9 : magnitude >= 0.0001 ? 10 : 12;
  return Number(number.toFixed(digits)).toLocaleString("zh-CN", {
    useGrouping: false, maximumFractionDigits: digits,
  });
}
function setStatus(el, text, kind) {
  el.textContent = text;
  el.classList.remove("status-ok", "status-warning", "status-stop");
  el.classList.add(kind);
}
async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
async function loadDatasets() {
  const payload = await fetchJson("/api/datasets");
  state.datasets = payload.datasets;
  const select = $("dataset");
  select.innerHTML = "";
  for (const item of state.datasets) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = `${datasetLabel(item)} (${item.rowCount} 行)`;
    select.appendChild(option);
  }
  if (state.datasets.some((item) => item.name === STRICT_DATASET)) {
    select.value = STRICT_DATASET;
  }
  renderColumnChoices();
  renderStaticHelp();
}
async function loadDeepLearningReport() {
  const report = await fetchJson("/api/deep-learning-report");
  state.latestDeepLearning = report;
  renderDeepLearning(report);
}
function selectedDataset() {
  return state.datasets.find((item) => item.name === $("dataset").value);
}
function renderColumnChoices() {
  const dataset = selectedDataset();
  const wrap = $("columns");
  wrap.innerHTML = "";
  $("sourcePill").textContent = sourceText(dataset);
  if (!dataset) return;
  const defaults = new Set(dataset.defaultColumns.length ? dataset.defaultColumns : dataset.columns.slice(1, 3));
  for (const column of dataset.columns) {
    if (column === "Time") continue;
    const meta = paramMeta(column);
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `column-${column}`;
    input.name = "columns";
    input.value = column;
    input.checked = defaults.has(column);
    input.setAttribute("aria-label", paramFullLabel(column));
    const text = document.createElement("span");
    text.className = "column-text";
    const name = document.createElement("span");
    name.className = "column-name";
    name.textContent = meta.label;
    const code = document.createElement("span");
    code.className = "column-code";
    code.textContent = column;
    text.append(name, code);
    label.title = meta.desc;
    label.append(input, text);
    wrap.appendChild(label);
    input.addEventListener("change", () => {
      renderParameterRules();
      if (state.latestDeepLearning) renderDeepLife(state.latestDeepLearning);
    });
  }
  renderParameterRules();
  renderColumnHelp(dataset);
  syncDeepLearningAvailability();
}
function syncDeepLearningAvailability() {
  const strictSelected = $("dataset")?.value === STRICT_DATASET;
  const tab = $("deepLearningTab");
  if (!tab) return;
  tab.disabled = !strictSelected;
  tab.title = strictSelected ? "" : "仅适用于实验波形严格特征数据集";
  const warning = $("deepDatasetWarning");
  if (warning) warning.hidden = strictSelected;
  if (!strictSelected && state.activePage === "deepLearningPage") {
    activateWorkspacePage("patentPage");
  }
}
function selectedColumns() {
  return [...document.querySelectorAll("#columns input:checked")].map((item) => item.value);
}
function collectParameterRules() {
  const rules = {};
  document.querySelectorAll("#parameterRules .rule-card").forEach((card) => {
    const column = card.dataset.column;
    if (!column) return;
    const direction = card.querySelector(".rule-direction")?.value || "increase";
    const percent = Number(card.querySelector(".rule-percent")?.value);
    if (Number.isFinite(percent) && percent >= 0) {
      rules[column] = { direction, percent };
    }
  });
  return rules;
}
function renderParameterRules() {
  const wrap = $("parameterRules");
  if (!wrap) return;
  const previous = collectParameterRules();
  const columns = selectedColumns();
  wrap.innerHTML = "";
  if (!columns.length) {
    const empty = document.createElement("div");
    empty.className = "rule-empty";
    empty.textContent = "请先选择参数列。";
    wrap.appendChild(empty);
    return;
  }
  for (const column of columns) {
    const meta = paramMeta(column);
    const rule = previous[column] || defaultRuleForColumn(column);
    const card = document.createElement("div");
    card.className = "rule-card";
    card.dataset.column = column;
    const heading = document.createElement("div");
    heading.className = "rule-heading";
    const name = document.createElement("div");
    name.className = "rule-name";
    name.textContent = meta.label;
    const code = document.createElement("div");
    code.className = "rule-code";
    code.textContent = column;
    heading.append(name, code);
    const controls = document.createElement("div");
    controls.className = "rule-controls";
    const select = document.createElement("select");
    select.className = "rule-direction";
    select.id = `rule-direction-${column}`;
    select.name = `rule-direction-${column}`;
    select.setAttribute("aria-label", `${meta.label}失效方向`);
    for (const [value, text] of [
      ["decrease", "下降失效"],
      ["increase", "上升失效"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      option.selected = rule.direction === value;
      select.appendChild(option);
    }
    const input = document.createElement("input");
    input.className = "rule-percent";
    input.type = "number";
    input.id = `rule-percent-${column}`;
    input.name = `rule-percent-${column}`;
    input.min = "0.1";
    input.max = "1000";
    input.step = "0.01";
    input.value = rule.percent;
    input.setAttribute("aria-label", `${meta.label}失效阈值百分比`);
    const suffix = document.createElement("span");
    suffix.className = "rule-suffix";
    suffix.textContent = "%";
    controls.append(select, input, suffix);
    card.append(heading, controls);
    wrap.appendChild(card);
    const updateDeepLife = () => {
      if (state.latestDeepLearning) renderDeepLife(state.latestDeepLearning);
    };
    select.addEventListener("change", updateDeepLife);
    input.addEventListener("change", updateDeepLife);
  }
}
function definitionItem(term, desc) {
  const item = document.createElement("div");
  item.className = "definition-item";
  const key = document.createElement("div");
  key.className = "definition-term";
  key.textContent = term;
  const value = document.createElement("div");
  value.className = "definition-desc";
  value.textContent = desc;
  item.append(key, value);
  return item;
}
function renderDefinitionList(id, entries) {
  const wrap = $(id);
  if (!wrap) return;
  wrap.innerHTML = "";
  for (const [term, desc] of entries) {
    wrap.appendChild(definitionItem(term, desc));
  }
}
function renderColumnHelp(dataset = selectedDataset()) {
  const wrap = $("columnHelp");
  if (!wrap || !dataset) return;
  wrap.innerHTML = "";
  for (const column of dataset.columns) {
    if (column === "Time") {
      wrap.appendChild(definitionItem("时间列（Time）", "每行数据对应的采样时间或相对时间，用于按时间换算预测步数。"));
      continue;
    }
    const meta = paramMeta(column);
    wrap.appendChild(definitionItem(`${meta.label}（${column}）`, meta.desc));
  }
}
function renderStaticHelp() {
  renderDefinitionList("inputHelp", INPUT_HELP);
  renderDefinitionList("outputHelp", OUTPUT_HELP);
  renderColumnHelp();
}
function paramBlock(column) {
  const wrap = document.createElement("div");
  wrap.className = "param-cell";
  const name = document.createElement("div");
  name.className = "param-name";
  name.textContent = paramLabel(column);
  const code = document.createElement("div");
  code.className = "param-code";
  code.textContent = column;
  wrap.append(name, code);
  return wrap;
}
function requestPayload() {
  const predictionByTime = selectedMode("predictionMode") === "time";
  const validationByTime = selectedMode("validationMode") === "time";
  return {
    dataset: $("dataset").value,
    columns: selectedColumns(),
    window: Number($("window").value),
    horizon: Number($("horizon").value),
    predictionTime: predictionByTime ? optionalNumber("predictionTime") : null,
    parameterRules: collectParameterRules(),
    accuracyThreshold: Number($("accuracy").value),
    lifeRatioThreshold: Number($("lifeRatio").value),
    validationHorizon: Number($("validationHorizon").value),
    validationTime: validationByTime ? optionalNumber("validationTime") : null,
    validationErrorThreshold: Number($("validationError").value),
    validationStartIndex: Number($("validationStart").value),
    validationEndIndex: optionalNumber("validationEnd"),
    validationStride: Number($("validationStride").value),
    smoothing: $("smoothing").checked,
  };
}
async function runEvaluation() {
  $("run").disabled = true;
  $("run").textContent = "计算中...";
  try {
    if (selectedMode("predictionMode") === "time" && optionalNumber("predictionTime") === null) {
      throw new Error("已选择按时间预测，请填写预测时长。");
    }
    if (selectedMode("validationMode") === "time" && optionalNumber("validationTime") === null) {
      throw new Error("已选择按时间验证，请填写验证时长。");
    }
    const payload = requestPayload();
    const [report, validation, modelComparison] = await Promise.all([
      fetchJson("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetchJson("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
      fetchJson("/api/compare-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    ]);
    state.latestEvaluation = report;
    state.latestValidation = validation;
    state.latestModelComparison = modelComparison;
    renderReport(report);
    renderValidation(validation);
    renderModelComparison(modelComparison);
  } catch (error) {
    const message =
      error instanceof TypeError && error.message.includes("fetch")
        ? "无法连接后端服务。请确认服务正在运行；本机打开建议使用 http://127.0.0.1:8890，内网电脑请使用服务器内网 IP，不要用 localhost。"
        : error.message;
    alert(message);
  } finally {
    $("run").disabled = false;
    $("run").textContent = "运行评估与验证";
  }
}
function renderReport(report) {
  const summary = report.summary;
  setStatus(
    $("engineeringStatus"),
    summary.engineeringStatus,
    summary.engineeringStatus === "正常" ? "status-ok" : "status-warning",
  );
  $("lifeRatioOut").textContent =
    summary.finalLifeRatioPercent === null ? "--" : `${fmt(summary.finalLifeRatioPercent, 2)}%`;
  $("accuracyOut").textContent = `${fmt(summary.finalAccuracyPercent, 2)}%`;
  $("lifeOut").textContent =
    summary.finalPredictedLifePeriods === null
      ? "未到临界"
      : `${fmt(summary.finalPredictedLifePeriods, 2)} 周期单位`;
  const rows = $("resultRows");
  rows.innerHTML = "";
  for (const item of report.results) {
    const tr = document.createElement("tr");
    const paramTd = document.createElement("td");
    paramTd.appendChild(paramBlock(item.column));
    tr.appendChild(paramTd);
    const cells = [
      item.direction === "increase" ? "上升失效" : "下降失效",
      item.baselineRow === 1
        ? fmtExact(item.baselineValue)
        : `${fmtExact(item.baselineValue)}（第 ${item.baselineRow} 行首个间隔）`,
      fmtThreshold(item.criticalValue),
      fmt(item.a, 5),
      fmt(item.b, 5),
      `${fmt(item.accuracyPercent, 2)}%`,
      item.levelRatioValid ? "通过" : `${item.levelRatioInvalidCount} 项异常`,
      item.remainingSamples === null ? "未预测到" : item.remainingSamples,
      item.engineeringStatus,
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    rows.appendChild(tr);
  }
  const notes = $("uncertainties");
  notes.innerHTML = "";
  for (const note of report.uncertainties) {
    const li = document.createElement("li");
    li.textContent = note;
    notes.appendChild(li);
  }
  drawChart(report);
}
function renderValidation(validation) {
  const summary = validation.summary;
  const status = $("validationStatus");
  status.textContent = summary.status;
  status.className = summary.status === "通过" ? "status-ok" : "status-warning";
  $("validationMape").textContent = `${fmt(summary.finalMapePercent, 2)}%`;
  $("validationWindow").textContent = validation.window;
  $("validationHorizonOut").textContent =
    validation.validationTime === null
      ? validation.horizon
      : `${validation.horizon} 步 / ${fmt(validation.validationTime, 2)} 时间`;
  $("validationRange").textContent = `${validation.validationStartIndex}-${validation.validationEndIndex}`;
  $("validationStrideOut").textContent = validation.validationStride;
  const rows = $("validationRows");
  rows.innerHTML = "";
  for (const item of validation.results) {
    const lastPoint = item.points[item.points.length - 1] || {};
    const metrics = item.metrics;
    const tr = document.createElement("tr");
    const paramTd = document.createElement("td");
    paramTd.appendChild(paramBlock(item.column));
    tr.appendChild(paramTd);
    const cells = [
      metrics.runs,
      fmt(metrics.mae, 5),
      fmt(metrics.rmse, 5),
      `${fmt(metrics.mapePercent, 2)}%`,
      `${fmt(metrics.maxAbsPercentError, 2)}%`,
      lastPoint.targetIndex || "--",
      fmt(lastPoint.predicted, 5),
      fmt(lastPoint.actual, 5),
      item.status,
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    rows.appendChild(tr);
  }
  drawValidationChart(validation);
}
function renderModelComparison(comparison) {
  const summary = comparison.summary;
  setStatus(
    $("bestModelStatus"),
    summary.status,
    summary.status === "通过" ? "status-ok" : "status-warning",
  );
  setStatus(
    $("bestValidationStatus"),
    summary.status,
    summary.status === "通过" ? "status-ok" : "status-warning",
  );
  $("bestModelAccuracy").textContent = `${fmt(summary.bestAccuracyPercent, 2)}%`;
  $("bestModelGmMape").textContent =
    summary.gmMapePercent === null ? "--" : `${fmt(summary.gmMapePercent, 2)}%`;
  const improvement = summary.mapeImprovementPoints;
  $("bestModelImprovement").textContent =
    improvement === null
      ? "--"
      : `${fmt(improvement, 2)} 个百分点${improvement < 0 ? "（未优于）" : ""}`;
  $("alternativeCount").textContent =
    `${summary.alternativeSelectionCount}/${comparison.results.length} 个参数选择非 GM 方法`;
  const bestRows = $("bestModelRows");
  bestRows.innerHTML = "";
  for (const item of comparison.results) {
    const lastPoint = item.validationPoints[item.validationPoints.length - 1] || {};
    const tr = document.createElement("tr");
    const paramTd = document.createElement("td");
    paramTd.appendChild(paramBlock(item.column));
    tr.appendChild(paramTd);
    const cells = [
      item.bestMethodName,
      `${fmt(item.bestMetrics.mapePercent, 2)}%`,
      item.gmMetrics === null ? "--" : `${fmt(item.gmMetrics.mapePercent, 2)}%`,
      item.mapeImprovementPoints === null ? "--" : fmt(item.mapeImprovementPoints, 2),
      fmt(lastPoint.predicted, 5),
      fmt(lastPoint.actual, 5),
      item.remainingSamples === null ? "未预测到" : item.remainingSamples,
      item.status,
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    bestRows.appendChild(tr);
  }
  const candidateRows = $("candidateModelRows");
  candidateRows.innerHTML = "";
  for (const item of comparison.results) {
    for (const candidate of item.candidateMetrics) {
      const tr = document.createElement("tr");
      if (candidate.method === item.bestMethod) tr.className = "selected-model-row";
      const paramTd = document.createElement("td");
      paramTd.appendChild(paramBlock(item.column));
      tr.appendChild(paramTd);
      const metrics = candidate.metrics;
      const cells = [
        candidate.methodName,
        `${fmt(metrics.mapePercent, 2)}%`,
        fmt(metrics.mae, 5),
        fmt(metrics.rmse, 5),
        `${fmt(metrics.maxAbsPercentError, 2)}%`,
        metrics.runs,
        candidate.method === item.bestMethod ? "已选择" : "候选",
      ];
      for (const cell of cells) {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      }
      candidateRows.appendChild(tr);
    }
  }
  renderDefinitionList(
    "methodHelp",
    comparison.methods.map((method) => [method.name, method.description]),
  );
  if (state.activePage === "alternativePage") {
    drawAlternativeChart(comparison);
    drawAlternativeValidationChart(comparison);
  }
}
function deepModelSeries(report, modelId) {
  const group = report.modelTestSeries?.find((item) => item.modelId === modelId);
  if (group) return group;
  if (modelId === report.bestTestModel && report.bestTestSeries) {
    return { modelId, modelName: report.bestTestModelName, series: report.bestTestSeries };
  }
  return {
    modelId: report.selectedDeepModel,
    modelName: report.selectedDeepModelName,
    series: report.selectedDeepTestSeries || [],
  };
}
function calculateDeepLife(report) {
  const forecast = report.lifeForecast;
  const maxAvailable = forecast.maxFutureSteps;
  const requestedLimit = Number($("deepLifeLimit")?.value) || maxAvailable;
  const limit = Math.max(20, Math.min(maxAvailable, Math.floor(requestedLimit)));
  const horizonAudit = forecast.horizonValidation || {};
  const validatedHorizon = Math.min(
    limit,
    Number(horizonAudit.validatedHorizonSteps) || Number(forecast.directHorizon) || 0,
  );
  const horizonResult = (horizonAudit.results || []).find(
    (item) => item.horizon === validatedHorizon,
  );
  const seriesByFeature = new Map(forecast.series.map((item) => [item.feature, item]));
  const diagnostics = new Map((forecast.diagnostics || []).map((item) => [item.feature, item]));
  const configuredRules = collectParameterRules();
  const selected = selectedColumns().filter((column) => seriesByFeature.has(column));
  const results = selected.map((feature) => {
    const diagnostic = diagnostics.get(feature) || {};
    const baseline = forecast.failureBaselineValues?.[feature]
      ?? forecast.healthyBaselineValues?.[feature]
      ?? forecast.currentValues[feature];
    const baselineRow = forecast.baselineRowsByFeature?.[feature] ?? 1;
    const current = forecast.robustCurrentValues?.[feature] ?? forecast.currentValues[feature];
    const rule = configuredRules[feature] || defaultRuleForColumn(feature);
    const critical = rule.direction === "decrease"
      ? baseline * (1 - rule.percent / 100)
      : baseline * (1 + rule.percent / 100);
    const values = seriesByFeature.get(feature).values.slice(0, limit);
    const crossingOffset = values.findIndex((value) =>
      rule.direction === "decrease" ? value <= critical : value >= critical,
    );
    const remaining = crossingOffset < 0 ? null : crossingOffset + 1;
    const span = Math.abs(critical - baseline);
    const directedChange = rule.direction === "decrease"
      ? baseline - current
      : current - baseline;
    const healthUsedPercent = span > 0
      ? Math.max(0, Math.min(100, directedChange / span * 100))
      : 0;
    const reliability = Math.max(0, Math.min(1, Number(diagnostic.reliabilityWeight) || 0));
    const trendWindows = diagnostic.trendWindows || [];
    const directedWindows = trendWindows.filter((item) =>
      item.trendDetectable && (rule.direction === "decrease"
        ? item.slopePerStep < 0
        : item.slopePerStep > 0),
    );
    const trendTowardFailure = directedWindows.length >= 2
      && directedWindows.some((item) => item.window <= 120);
    const validatedRemaining = remaining !== null
      && remaining <= validatedHorizon
      && reliability >= 0.35
      && trendTowardFailure
      ? remaining
      : null;
    return {
      feature, direction: rule.direction, percent: rule.percent, baseline, baselineRow, current,
      critical, terminal: values[values.length - 1], remaining, validatedRemaining,
      reliability, healthUsedPercent, trendTowardFailure,
      recentUniqueLevels: diagnostic.recentUniqueLevels,
      testNmaePercent: diagnostic.testNmaePercent,
    };
  });
  const validatedCrossings = results.filter((item) => item.validatedRemaining !== null);
  const limiting = validatedCrossings.length
    ? validatedCrossings.reduce((best, item) =>
      item.validatedRemaining < best.validatedRemaining ? item : best)
    : null;
  const exploratoryCrossings = results.filter((item) => item.remaining !== null);
  const exploratoryLimiter = exploratoryCrossings.length
    ? exploratoryCrossings.reduce((best, item) => item.remaining < best.remaining ? item : best)
    : null;
  const weightSum = results.reduce((sum, item) => sum + item.reliability, 0);
  const weightedHealthUsedPercent = weightSum
    ? results.reduce((sum, item) => sum + item.healthUsedPercent * item.reliability, 0) / weightSum
    : null;
  const confidencePercent = results.length ? weightSum / results.length * 100 : 0;
  const periodSeconds = Number(forecast.medianPeriodSeconds) || null;
  const remainingSteps = limiting?.validatedRemaining ?? validatedHorizon;
  return {
    forecast, limit, validatedHorizon, results, limiting, exploratoryLimiter,
    totalLife: limiting ? forecast.currentRow + limiting.validatedRemaining : null,
    validatedTotalLowerBound: forecast.currentRow + validatedHorizon,
    validatedAccuracyPercent: horizonResult?.accuracyPercent ?? null,
    weightedHealthUsedPercent,
    healthScorePercent: weightedHealthUsedPercent === null ? null : 100 - weightedHealthUsedPercent,
    confidencePercent, periodSeconds,
    remainingTimeSeconds: periodSeconds ? remainingSteps * periodSeconds : null,
    exploratoryTimeSeconds: periodSeconds ? limit * periodSeconds : null,
    identifiable: Boolean(limiting),
  };
}
function renderDeepLife(report) {
  if (!report.lifeForecast) return;
  const limitInput = $("deepLifeLimit");
  limitInput.max = String(report.lifeForecast.maxFutureSteps);
  limitInput.value = String(Math.max(20, Math.min(
    report.lifeForecast.maxFutureSteps,
    Math.floor(Number(limitInput.value) || report.lifeForecast.maxFutureSteps),
  )));
  const life = calculateDeepLife(report);
  const featureSelect = $("deepLifeFeatureSelect");
  const previousFeature = featureSelect.value;
  featureSelect.innerHTML = "";
  for (const result of life.results) {
    const option = document.createElement("option");
    option.value = result.feature;
    option.textContent = paramLabel(result.feature);
    featureSelect.appendChild(option);
  }
  if (life.results.length) {
    featureSelect.value = life.results.some((item) => item.feature === previousFeature)
      ? previousFeature
      : life.limiting?.feature || life.results[0].feature;
  }
  featureSelect.disabled = !life.results.length;
  const duration = (seconds) => {
    if (seconds === null) return "";
    return seconds >= 3600
      ? `约 ${fmt(seconds / 3600, 2)} 小时`
      : `约 ${fmt(seconds / 60, 1)} 分钟`;
  };
  $("deepLifeBasis").textContent =
    `阈值基准：${life.forecast.baselineDefinition || "原始数据第 1 次放电"}；` +
    (life.forecast.failureCalibration
      ? `实测标定：第 ${life.forecast.failureCalibration.totalFailureRow} 次失效；`
      : "") +
    `回测支持 ${life.validatedHorizon} 步（精度 ${fmt(life.validatedAccuracyPercent, 2)}%）。`;
  const calibration = life.forecast.failureCalibration;
  $("deepLifeCalibration").textContent = calibration
    ? `剩余 ${calibration.additionalFailureSteps} 步 / 总 ${calibration.totalFailureRow} 次`
    : "--";
  $("deepLifeConfidence").textContent = `${fmt(life.confidencePercent, 1)}%`;
  $("deepLifeExploratory").textContent = life.exploratoryLimiter
    ? `${life.exploratoryLimiter.remaining} 步达到阈值（未验证）`
    : `${life.limit} 步内未达阈值（未验证）`;
  if (!life.results.length) {
    setStatus($("deepLifeStatus"), "请在左侧勾选严格特征", "status-warning");
    for (const id of ["deepLifeRemaining", "deepLifeTotal", "deepLifeRatio", "deepLifeLimiter"]) {
      $(id).textContent = "--";
    }
  } else if (life.limiting) {
    setStatus($("deepLifeStatus"), "有限寿命可辨识", "status-warning");
    $("deepLifeRemaining").textContent =
      `${life.limiting.validatedRemaining} 步（${duration(life.remainingTimeSeconds)}）`;
    $("deepLifeTotal").textContent = `${life.totalLife} 步`;
    $("deepLifeRatio").textContent = `${fmt(life.healthScorePercent, 1)}%`;
    $("deepLifeLimiter").textContent = paramLabel(life.limiting.feature);
  } else {
    setStatus($("deepLifeStatus"), "有限寿命暂不可辨识", "status-ok");
    $("deepLifeRemaining").textContent =
      `>${life.validatedHorizon} 步（经验证，${duration(life.remainingTimeSeconds)}）`;
    $("deepLifeTotal").textContent = `>${life.validatedTotalLowerBound} 步（经验证）`;
    $("deepLifeRatio").textContent = `${fmt(life.healthScorePercent, 1)}%`;
    $("deepLifeLimiter").textContent = "无可信限寿参数";
  }
  const rows = $("deepLifeRows");
  rows.innerHTML = "";
  for (const result of life.results) {
    const tr = document.createElement("tr");
    if (result.feature === life.limiting?.feature) tr.className = "selected-model-row";
    const parameter = document.createElement("td");
    parameter.appendChild(paramBlock(result.feature));
    tr.appendChild(parameter);
    let conclusion = "验证范围未到临界";
    if (result.reliability < 0.35) conclusion = "可靠度不足";
    else if (!result.trendTowardFailure) conclusion = "无一致失效趋势";
    else if (result.remaining !== null && result.remaining > life.validatedHorizon) {
      conclusion = "仅外推达到";
    } else if (result.validatedRemaining !== null) conclusion = "有限寿命可辨识";
    const cells = [
      result.direction === "decrease" ? "下降失效" : "上升失效",
      result.baselineRow === 1
        ? `${fmtExact(result.baseline)}（第 1 次放电）`
        : `${fmtExact(result.baseline)}（第 ${result.baselineRow} 行首个间隔）`,
      fmtExact(result.current), fmtThreshold(result.critical),
      `${fmt(result.reliability * 100, 1)}%`,
      result.validatedRemaining !== null
        ? `${result.validatedRemaining} 步`
        : `>${life.validatedHorizon} 步（经验证）`,
      conclusion,
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    rows.appendChild(tr);
  }
  if (state.activePage === "deepLearningPage" && life.results.length) {
    drawDeepLifeChart(report, life);
  }
}
function renderDeepLearning(report) {
  const summary = report.summary;
  const rankedModels = [...report.models].sort(
    (left, right) =>
      left.testMetrics.meanNmaePercent - right.testMetrics.meanNmaePercent,
  );
  const bestModel =
    rankedModels.find((model) => model.id === report.bestTestModel) || rankedModels[0];
  const bestAccuracy =
    summary.bestTestNormalizedAccuracyPercent ?? bestModel.testMetrics.normalizedAccuracyPercent;
  const bestNmae = summary.bestTestNmaePercent ?? bestModel.testMetrics.meanNmaePercent;
  $("deepSelectedModel").textContent = report.selectedModelName || report.selectedDeepModelName;
  $("deepBestTestModel").textContent = bestModel.name;
  $("deepBestLookback").textContent = `${bestModel.lookback || report.lookback} 步`;
  $("deepTestAccuracy").textContent = `${fmt(bestAccuracy, 2)}%`;
  $("deepTestNmae").textContent = `${fmt(bestNmae, 2)}%`;
  setStatus(
    $("deepTestConclusion"),
    `独立测试最高：${bestModel.name} / ${bestModel.lookback || report.lookback} 步`,
    "status-ok",
  );
  $("deepLookback").textContent = `${report.lookback} 步`;
  $("deepHorizon").textContent = `${report.horizon} 步`;
  $("deepTrainRows").textContent = `1-${report.split.trainEndRow}`;
  $("deepValidationRows").textContent =
    `${report.split.validationStartRow}-${report.split.validationEndRow}`;
  $("deepTestRows").textContent = `${report.split.testStartRow}-${report.split.testEndRow}`;
  $("deepFeatureCount").textContent = `${report.features.length} 个严格特征 / ${bestModel.name}`;
  $("deepDevice").textContent = `${report.environment.deviceName} / ${fmt(report.environment.trainingSeconds, 2)} 秒`;
  const lookbackSearch = report.lookbackSearch;
  const lookbackRows = $("deepLookbackRows");
  lookbackRows.innerHTML = "";
  if (lookbackSearch) {
    $("deepLookbackSearchStatus").textContent =
      `${lookbackSearch.bestModelName} / ${lookbackSearch.candidates.length} 组候选 / ` +
      `最佳 ${lookbackSearch.bestLookback} 步`;
    for (const candidate of lookbackSearch.candidates) {
      const tr = document.createElement("tr");
      if (candidate.lookback === lookbackSearch.bestLookback) {
        tr.className = "selected-model-row";
      }
      const cells = [
        `${candidate.lookback} 步`,
        candidate.focusModelName || lookbackSearch.bestModelName,
        `${fmt(candidate.validationNmaePercent, 2)}%`,
        `${fmt(candidate.testNmaePercent, 2)}%`,
        candidate.lookback === lookbackSearch.bestLookback ? "验证集选中" : "候选",
      ];
      for (const cell of cells) {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      }
      lookbackRows.appendChild(tr);
    }
  } else {
    $("deepLookbackSearchStatus").textContent = "未执行窗口搜索";
  }
  const modelSelect = $("deepModelSelect");
  const previousModel = modelSelect.value;
  modelSelect.innerHTML = "";
  for (const model of rankedModels) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent =
      `${model.name}（${model.lookback || report.lookback} 步，测试精度 ` +
      `${fmt(model.testMetrics.normalizedAccuracyPercent, 2)}%）`;
    modelSelect.appendChild(option);
  }
  modelSelect.value = report.models.some((model) => model.id === previousModel)
    ? previousModel
    : bestModel.id;
  const featureSelect = $("deepFeatureSelect");
  const previousFeature = featureSelect.value;
  featureSelect.innerHTML = "";
  for (const feature of report.features) {
    const option = document.createElement("option");
    option.value = feature.feature;
    option.textContent = feature.label;
    featureSelect.appendChild(option);
  }
  featureSelect.value = report.features.some((feature) => feature.feature === previousFeature)
    ? previousFeature
    : report.features[0].feature;
  const bestSeries = deepModelSeries(report, bestModel.id).series;
  const metricsByFeature = new Map(
    bestModel.testMetrics.perFeature.map((metrics) => [metrics.feature, metrics]),
  );
  const seriesByFeature = new Map(
    bestSeries.map((series) => [series.feature, series]),
  );
  const featureRows = $("deepFeatureRows");
  featureRows.innerHTML = "";
  for (const feature of report.features) {
    const metrics = metricsByFeature.get(feature.feature);
    const series = seriesByFeature.get(feature.feature);
    const lastPoint = series?.points?.[series.points.length - 1] || {};
    const tr = document.createElement("tr");
    const paramTd = document.createElement("td");
    paramTd.appendChild(paramBlock(feature.feature));
    tr.appendChild(paramTd);
    const cells = [
      fmt(metrics.mae, 6),
      fmt(metrics.rmse, 6),
      `${fmt(metrics.nmaePercent, 2)}%`,
      `${fmt(metrics.smapePercent, 2)}%`,
      `${fmt(metrics.mapePercent, 2)}%`,
      fmt(lastPoint.predicted, 6),
      fmt(lastPoint.actual, 6),
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    featureRows.appendChild(tr);
  }
  const modelRows = $("deepModelRows");
  modelRows.innerHTML = "";
  for (const [rank, model] of rankedModels.entries()) {
    const tr = document.createElement("tr");
    if (model.id === report.bestTestModel) tr.className = "selected-model-row";
    const selectionLabels = [];
    if (model.id === report.selectedModel) selectionLabels.push("验证集选中模型");
    if (model.id === report.selectedDeepModel) selectionLabels.push("验证集选中深度模型");
    if (model.id === report.bestTestModel) selectionLabels.push("独立测试最高");
    if (model.id === report.selectedBaseline) selectionLabels.push("最佳基线");
    const typeLabels = {
      patent: "专利模型",
      statistical: "统计模型",
      baseline: "基线",
      hybrid: "量化感知混合模型",
      deep_learning: "深度学习",
    };
    const cells = [
      String(rank + 1),
      model.name,
      typeLabels[model.modelType] || model.modelType,
      `${model.lookback || report.lookback} 步`,
      `${fmt(model.validationMetrics.meanNmaePercent, 2)}%`,
      `${fmt(model.testMetrics.meanNmaePercent, 2)}%`,
      `${fmt(model.testMetrics.normalizedAccuracyPercent, 2)}%`,
      `${fmt(model.testMetrics.meanSmapePercent, 2)}%`,
      fmt(model.trainingSeconds, 2),
      selectionLabels.join(" / ") || "候选",
    ];
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    modelRows.appendChild(tr);
  }
  const notes = $("deepNotes");
  notes.innerHTML = "";
  for (const note of report.notes) {
    const li = document.createElement("li");
    li.textContent = note;
    notes.appendChild(li);
  }
  renderDeepLife(report);
  syncDeepLearningAvailability();
  if (state.activePage === "deepLearningPage") {
    drawDeepPredictionChart(report);
    drawDeepLearningChart(report);
  }
}
function scaleFor(seriesList, width, height, pad) {
  const values = seriesList.flatMap((series) => series.values.filter(Number.isFinite));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const maxLen = Math.max(...seriesList.map((series) => series.values.length));
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  return {
    x: (idx) => pad.left + (idx / Math.max(1, maxLen - 1)) * plotWidth,
    y: (value) => height - pad.bottom - ((value - min) / span) * plotHeight,
    inverseX: (pixel) => ((pixel - pad.left) / plotWidth) * Math.max(1, maxLen - 1),
    inverseY: (pixel) => min + ((height - pad.bottom - pixel) / plotHeight) * span,
    min,
    max,
    minX: 0,
    maxX: Math.max(1, maxLen - 1),
  };
}
function scaleForXY(seriesList, width, height, pad) {
  const points = seriesList.flatMap((series) => series.points);
  const xs = points.map((point) => point.x).filter(Number.isFinite);
  const ys = points.map((point) => point.y).filter(Number.isFinite);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  return {
    x: (value) => pad.left + ((value - minX) / spanX) * plotWidth,
    y: (value) => height - pad.bottom - ((value - minY) / spanY) * plotHeight,
    inverseX: (pixel) => minX + ((pixel - pad.left) / plotWidth) * spanX,
    inverseY: (pixel) => minY + ((height - pad.bottom - pixel) / plotHeight) * spanY,
    minX,
    maxX,
    min: minY,
    max: maxY,
  };
}
function drawAxisLabels(ctx, width, height, pad, xLabel, yLabel) {
  ctx.save();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  const x = pad.left + (width - pad.left - pad.right) / 2;
  ctx.fillText(xLabel, x - ctx.measureText(xLabel).width / 2, height - 14);
  ctx.translate(18, pad.top + (height - pad.top - pad.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, -ctx.measureText(yLabel).width / 2, 0);
  ctx.restore();
}
function drawLine(ctx, values, scale, color, width = 2, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  values.forEach((value, idx) => {
    const x = scale.x(idx);
    const y = scale.y(value);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}
function drawLineXY(ctx, points, scale, color, width = 2, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, idx) => {
    const x = scale.x(point.x);
    const y = scale.y(point.y);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}
const chartHoverBindings = new WeakMap();
function hoverNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (Math.abs(number) >= 10000 || (Math.abs(number) > 0 && Math.abs(number) < 0.0001)) {
    return number.toExponential(6);
  }
  return number.toFixed(6).replace(/\.?0+$/, "");
}
function registerChartHover(canvas, config) {
  if (!canvas || !config.series.some((series) => series.points.length)) return;
  let binding = chartHoverBindings.get(canvas);
  if (!binding) {
    const host = canvas.closest(".chart-wrap, .validation-panel") || canvas.parentElement;
    host.classList.add("interactive-chart-host");
    const layer = document.createElement("div");
    layer.className = "chart-hover-layer";
    layer.hidden = true;
    const vertical = document.createElement("div");
    vertical.className = "chart-crosshair chart-crosshair-vertical";
    const horizontal = document.createElement("div");
    horizontal.className = "chart-crosshair chart-crosshair-horizontal";
    const tooltip = document.createElement("div");
    tooltip.className = "chart-hover-tooltip";
    layer.append(vertical, horizontal, tooltip);
    host.appendChild(layer);
    binding = { host, layer, vertical, horizontal, tooltip, config };
    chartHoverBindings.set(canvas, binding);
    const hide = () => {
      binding.layer.hidden = true;
    };
    canvas.addEventListener("pointerleave", hide);
    canvas.addEventListener("pointermove", (event) => {
      const current = binding.config;
      const canvasRect = canvas.getBoundingClientRect();
      const hostRect = binding.host.getBoundingClientRect();
      const localX = event.clientX - canvasRect.left;
      const localY = event.clientY - canvasRect.top;
      const pixelRatio = window.devicePixelRatio || 1;
      const logicalWidth = canvas.width / pixelRatio;
      const logicalHeight = canvas.height / pixelRatio;
      const logicalX = localX * (logicalWidth / canvasRect.width);
      const logicalY = localY * (logicalHeight / canvasRect.height);
      const { pad, scale } = current;
      if (
        logicalX < pad.left ||
        logicalX > logicalWidth - pad.right ||
        logicalY < pad.top ||
        logicalY > logicalHeight - pad.bottom
      ) {
        hide();
        return;
      }
      const targetX = scale.inverseX(logicalX);
      const referencePoints = current.series.flatMap((series) => series.points);
      const nearest = referencePoints.reduce((best, point) =>
        Math.abs(point.x - targetX) < Math.abs(best.x - targetX) ? point : best,
      );
      const exactX = nearest.x;
      const values = current.series
        .map((series) => {
          if (!series.points.length) return null;
          const xs = series.points.map((point) => point.x);
          const minSeriesX = Math.min(...xs);
          const maxSeriesX = Math.max(...xs);
          if (!series.continuous && (exactX < minSeriesX || exactX > maxSeriesX)) {
            return null;
          }
          const point = series.points.reduce((best, candidate) =>
            Math.abs(candidate.x - exactX) < Math.abs(best.x - exactX) ? candidate : best,
          );
          if (!series.continuous && Math.abs(point.x - exactX) > 1e-6) return null;
          return { ...series, point };
        })
        .filter(Boolean);
      if (!values.length) {
        hide();
        return;
      }
      const snappedX = scale.x(exactX) * (canvasRect.width / logicalWidth);
      const snappedY = scale.y(values[0].point.y) * (canvasRect.height / logicalHeight);
      const displayPadLeft = pad.left * (canvasRect.width / logicalWidth);
      const displayPadRight = pad.right * (canvasRect.width / logicalWidth);
      const displayPadTop = pad.top * (canvasRect.height / logicalHeight);
      const displayPadBottom = pad.bottom * (canvasRect.height / logicalHeight);
      const canvasLeft = canvasRect.left - hostRect.left;
      const canvasTop = canvasRect.top - hostRect.top;
      binding.vertical.style.left = `${canvasLeft + snappedX}px`;
      binding.vertical.style.top = `${canvasTop + displayPadTop}px`;
      binding.vertical.style.height = `${canvasRect.height - displayPadTop - displayPadBottom}px`;
      binding.horizontal.style.left = `${canvasLeft + displayPadLeft}px`;
      binding.horizontal.style.top = `${canvasTop + snappedY}px`;
      binding.horizontal.style.width = `${canvasRect.width - displayPadLeft - displayPadRight}px`;
      binding.tooltip.innerHTML = "";
      const heading = document.createElement("div");
      heading.className = "chart-tooltip-heading";
      heading.textContent = current.xFormatter(exactX);
      binding.tooltip.appendChild(heading);
      for (const value of values) {
        const row = document.createElement("div");
        row.className = "chart-tooltip-row";
        const marker = document.createElement("span");
        marker.className = "chart-tooltip-marker";
        marker.style.background = value.color;
        const label = document.createElement("span");
        label.textContent = value.label;
        const number = document.createElement("strong");
        number.textContent = hoverNumber(value.point.y);
        row.append(marker, label, number);
        binding.tooltip.appendChild(row);
      }
      const tooltipWidth = 260;
      const desiredLeft = canvasLeft + snappedX + 12;
      const desiredTop = canvasTop + snappedY + 12;
      binding.layer.hidden = false;
      binding.tooltip.style.left = `${Math.max(8, Math.min(desiredLeft, binding.host.clientWidth - tooltipWidth - 8))}px`;
      binding.tooltip.style.top = `${Math.max(8, Math.min(desiredTop, binding.host.clientHeight - binding.tooltip.offsetHeight - 8))}px`;
    });
  }
  binding.config = config;
}
function drawChart(report) {
  const canvas = $("chart");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(800, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 72, right: 24, top: 24, bottom: 54 };
  const seriesList = report.results.map((item) => ({
    name: paramLabel(item.column),
    values: item.forecast,
  }));
  if (!seriesList.length) return;
  const scale = scaleFor(seriesList, width, height, pad);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(fmt(scale.max, 3), 36, pad.top + 4);
  ctx.fillText(fmt(scale.min, 3), 36, height - pad.bottom);
  ctx.fillText("1", pad.left, height - pad.bottom + 18);
  ctx.fillText(String(Math.max(...seriesList.map((series) => series.values.length))), width - pad.right - 28, height - pad.bottom + 18);
  drawAxisLabels(ctx, width, height, pad, "横轴：样本序号（实测段 + 预测段）", "纵轴：参数值");
  report.results.forEach((item, idx) => {
    const color = palette[idx % palette.length];
    drawLine(ctx, item.forecast, scale, color, 2, [6, 5]);
    drawLine(ctx, item.series, scale, color, 3, []);
    const legendX = Math.max(pad.left + 20, width - 360);
    ctx.fillStyle = color;
    ctx.fillRect(legendX, pad.top + idx * 22 - 9, 14, 4);
    ctx.fillText(`${paramLabel(item.column)} 实测/预测`, legendX + 22, pad.top + idx * 22);
  });
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `样本序号：${Math.round(value) + 1}`,
    series: report.results.flatMap((item, idx) => {
      const color = palette[idx % palette.length];
      return [
        {
          label: `${paramLabel(item.column)}真实值`,
          color,
          points: item.series.map((value, index) => ({ x: index, y: value })),
        },
        {
          label: `${paramLabel(item.column)}预测值`,
          color: "#2764a8",
          points: item.forecast.map((value, index) => ({ x: index, y: value })),
        },
      ];
    }),
  });
}
function drawAlternativeChart(comparison) {
  const canvas = $("alternativeChart");
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(800, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 72, right: 24, top: 24, bottom: 54 };
  const seriesList = comparison.results.map((item) => ({
    name: paramLabel(item.column),
    values: item.forecast,
  }));
  if (!seriesList.length) return;
  const scale = scaleFor(seriesList, width, height, pad);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(fmt(scale.max, 3), 36, pad.top + 4);
  ctx.fillText(fmt(scale.min, 3), 36, height - pad.bottom);
  ctx.fillText("1", pad.left, height - pad.bottom + 18);
  ctx.fillText(
    String(Math.max(...seriesList.map((series) => series.values.length))),
    width - pad.right - 28,
    height - pad.bottom + 18,
  );
  drawAxisLabels(ctx, width, height, pad, "横轴：样本序号（实测段 + 预测段）", "纵轴：参数值");
  comparison.results.forEach((item, idx) => {
    const color = palette[idx % palette.length];
    drawLine(ctx, item.forecast, scale, color, 2, [6, 5]);
    drawLine(ctx, item.series, scale, color, 3, []);
    const legendX = Math.max(pad.left + 20, width - 430);
    ctx.fillStyle = color;
    ctx.fillRect(legendX, pad.top + idx * 22 - 9, 14, 4);
    ctx.fillText(
      `${paramLabel(item.column)}：${item.bestMethodName}`,
      legendX + 22,
      pad.top + idx * 22,
    );
  });
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `样本序号：${Math.round(value) + 1}`,
    series: comparison.results.flatMap((item, idx) => {
      const color = palette[idx % palette.length];
      return [
        {
          label: `${paramLabel(item.column)}真实值`,
          color,
          points: item.series.map((value, index) => ({ x: index, y: value })),
        },
        {
          label: `${paramLabel(item.column)}预测值`,
          color: "#2764a8",
          points: item.forecast.map((value, index) => ({ x: index, y: value })),
        },
      ];
    }),
  });
}
function drawValidationChart(validation) {
  const canvas = $("validationChart");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(800, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 72, right: 24, top: 22, bottom: 54 };
  const seriesList = validation.results.map((item) => ({
    name: paramLabel(item.column),
    points: item.points.map((point) => ({
      x: point.targetIndex,
      y: point.absPercentError,
    })),
  }));
  if (!seriesList.length || !seriesList.some((series) => series.points.length)) return;
  const scale = scaleForXY(seriesList, width, height, pad);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(`${fmt(scale.max, 2)}%`, 36, pad.top + 4);
  ctx.fillText(`${fmt(scale.min, 2)}%`, 36, height - pad.bottom);
  ctx.fillText(String(Math.round(scale.minX)), pad.left, height - pad.bottom + 18);
  ctx.fillText(String(Math.round(scale.maxX)), width - pad.right - 38, height - pad.bottom + 18);
  drawAxisLabels(ctx, width, height, pad, "横轴：真实对比行号", "纵轴：绝对百分比误差 (%)");
  validation.results.forEach((item, idx) => {
    const color = palette[idx % palette.length];
    const points = item.points.map((point) => ({
      x: point.targetIndex,
      y: point.absPercentError,
    }));
    drawLineXY(ctx, points, scale, color, 2, []);
    const legendX = Math.max(pad.left + 20, width - 360);
    ctx.fillStyle = color;
    ctx.fillRect(legendX, pad.top + idx * 22 - 9, 14, 4);
    ctx.fillText(`${paramLabel(item.column)} 验证误差`, legendX + 22, pad.top + idx * 22);
  });
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `真实对比行：${Math.round(value)}`,
    series: validation.results.map((item, idx) => ({
      label: `${paramLabel(item.column)}误差 (%)`,
      color: palette[idx % palette.length],
      points: item.points.map((point) => ({
        x: point.targetIndex,
        y: point.absPercentError,
      })),
    })),
  });
}
function drawAlternativeValidationChart(comparison) {
  const canvas = $("alternativeValidationChart");
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(800, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 72, right: 24, top: 22, bottom: 54 };
  const seriesList = comparison.results.map((item) => ({
    name: paramLabel(item.column),
    points: item.validationPoints.map((point) => ({
      x: point.targetIndex,
      y: point.absPercentError,
    })),
  }));
  if (!seriesList.length || !seriesList.some((series) => series.points.length)) return;
  const scale = scaleForXY(seriesList, width, height, pad);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(`${fmt(scale.max, 2)}%`, 36, pad.top + 4);
  ctx.fillText(`${fmt(scale.min, 2)}%`, 36, height - pad.bottom);
  ctx.fillText(String(Math.round(scale.minX)), pad.left, height - pad.bottom + 18);
  ctx.fillText(String(Math.round(scale.maxX)), width - pad.right - 38, height - pad.bottom + 18);
  drawAxisLabels(ctx, width, height, pad, "横轴：真实对比行号", "纵轴：绝对百分比误差 (%)");
  comparison.results.forEach((item, idx) => {
    const color = palette[idx % palette.length];
    const points = item.validationPoints.map((point) => ({
      x: point.targetIndex,
      y: point.absPercentError,
    }));
    drawLineXY(ctx, points, scale, color, 2, []);
    const legendX = Math.max(pad.left + 20, width - 430);
    ctx.fillStyle = color;
    ctx.fillRect(legendX, pad.top + idx * 22 - 9, 14, 4);
    ctx.fillText(
      `${paramLabel(item.column)}：${item.bestMethodName}`,
      legendX + 22,
      pad.top + idx * 22,
    );
  });
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `真实对比行：${Math.round(value)}`,
    series: comparison.results.map((item, idx) => ({
      label: `${paramLabel(item.column)}误差 (%)`,
      color: palette[idx % palette.length],
      points: item.validationPoints.map((point) => ({
        x: point.targetIndex,
        y: point.absPercentError,
      })),
    })),
  });
}
function drawDeepPredictionChart(report) {
  const canvas = $("deepPredictionChart");
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const modelId = $("deepModelSelect")?.value || report.bestTestModel;
  const featureId = $("deepFeatureSelect")?.value || report.features[0].feature;
  const group = deepModelSeries(report, modelId);
  const series = group.series.find((item) => item.feature === featureId);
  if (!series?.points?.length) return;
  $("deepPredictionTitle").textContent = `${group.modelName} / ${paramLabel(featureId)}`;
  const actualPoints = series.points.map((point) => ({
    x: point.targetIndex,
    y: point.actual,
  }));
  const predictedPoints = series.points.map((point) => ({
    x: point.targetIndex,
    y: point.predicted,
  }));
  const pad = { left: 88, right: 24, top: width < 600 ? 48 : 34, bottom: 54 };
  const scale = scaleForXY(
    [{ points: actualPoints }, { points: predictedPoints }],
    width,
    height,
    pad,
  );
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(fmt(scale.max, 4), 28, pad.top + 4);
  ctx.fillText(fmt(scale.min, 4), 28, height - pad.bottom);
  ctx.fillText(String(Math.round(scale.minX)), pad.left, height - pad.bottom + 18);
  ctx.fillText(String(Math.round(scale.maxX)), width - pad.right - 38, height - pad.bottom + 18);
  drawAxisLabels(
    ctx,
    width,
    height,
    pad,
    "横轴：独立测试目标行号",
    `纵轴：${paramLabel(featureId)}参数值`,
  );
  drawLineXY(ctx, actualPoints, scale, "#0e766d", 3, []);
  drawLineXY(ctx, predictedPoints, scale, "#2764a8", 2, [7, 5]);
  const legendX = Math.max(pad.left + 8, width - 196);
  ctx.strokeStyle = "#0e766d";
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(legendX, 15);
  ctx.lineTo(legendX + 22, 15);
  ctx.stroke();
  ctx.fillStyle = "#182021";
  ctx.fillText("真实值", legendX + 30, 19);
  ctx.strokeStyle = "#2764a8";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  ctx.beginPath();
  ctx.moveTo(legendX + 94, 15);
  ctx.lineTo(legendX + 116, 15);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillText("预测值", legendX + 124, 19);
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `独立测试目标行：${Math.round(value)}`,
    series: [
      { label: "真实值", color: "#0e766d", points: actualPoints },
      { label: `${group.modelName}预测值`, color: "#2764a8", points: predictedPoints },
    ],
  });
}
function drawDeepLifeChart(report, life = calculateDeepLife(report)) {
  const canvas = $("deepLifeChart");
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || !life.results.length) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const featureId = $("deepLifeFeatureSelect").value || life.results[0].feature;
  const featureSeries = life.forecast.series.find((item) => item.feature === featureId);
  const featureResult = life.results.find((item) => item.feature === featureId);
  if (!featureSeries || !featureResult) return;
  const observedPoints = featureSeries.observedValues.map((value, index, values) => ({
    x: index - values.length + 1,
    y: value,
  }));
  const futurePoints = [
    { x: 0, y: featureResult.current },
    ...featureSeries.values.slice(0, life.limit).map((value, index) => ({
      x: index + 1,
      y: value,
    })),
  ];
  const validatedPoints = futurePoints.filter((point) => point.x <= life.validatedHorizon);
  const exploratoryPoints = futurePoints.filter((point) => point.x >= life.validatedHorizon);
  const thresholdPoints = [
    { x: 0, y: featureResult.critical },
    { x: life.limit, y: featureResult.critical },
  ];
  const pad = { left: 88, right: 24, top: 34, bottom: 54 };
  const scale = scaleForXY(
    [{ points: observedPoints }, { points: futurePoints }, { points: thresholdPoints }],
    width,
    height,
    pad,
  );
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.save();
  ctx.strokeStyle = "#9aa4a1";
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(scale.x(0), pad.top);
  ctx.lineTo(scale.x(0), height - pad.bottom);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(fmt(scale.max, 4), 28, pad.top + 4);
  ctx.fillText(fmt(scale.min, 4), 28, height - pad.bottom);
  ctx.fillText(String(Math.round(scale.minX)), pad.left, height - pad.bottom + 18);
  if (scale.x(0) - pad.left > 36) ctx.fillText("0", scale.x(0) - 4, height - pad.bottom + 18);
  ctx.fillText(String(life.limit), width - pad.right - 38, height - pad.bottom + 18);
  drawAxisLabels(
    ctx,
    width,
    height,
    pad,
    `横轴：相对当前行步数（0 为第 ${life.forecast.currentRow} 行）`,
    `纵轴：${paramLabel(featureId)}参数值`,
  );
  drawLineXY(ctx, observedPoints, scale, "#0e766d", 3, []);
  drawLineXY(ctx, validatedPoints, scale, "#2764a8", 2, [7, 5]);
  drawLineXY(ctx, exploratoryPoints, scale, "#8a6b35", 2, [2, 6]);
  drawLineXY(ctx, thresholdPoints, scale, "#aa2b2b", 2, [4, 4]);
  const compactLegend = width < 600;
  const legendColumns = compactLegend ? 2 : 4;
  const legendX = compactLegend ? pad.left : Math.max(pad.left + 8, width - 360);
  const legendGap = compactLegend ? (width - pad.left - pad.right) / 2 : 90;
  for (const [index, item] of [
    ["#0e766d", "实测", []], ["#2764a8", "验证预测", [7, 5]],
    ["#8a6b35", "探索外推", [2, 6]], ["#aa2b2b", "阈值", [4, 4]],
  ].entries()) {
    const [color, label, dash] = item;
    const x = legendX + (index % legendColumns) * legendGap;
    const y = 15 + Math.floor(index / legendColumns) * 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 18, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#182021";
    ctx.fillText(label, x + 23, y + 4);
  }
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) =>
      value < 0
        ? `当前行之前：${Math.abs(Math.round(value))} 步`
        : `相对第 ${life.forecast.currentRow} 行：+${Math.round(value)} 步`,
    series: [
      { label: "最近实测值", color: "#0e766d", points: observedPoints },
      { label: "验证范围预测值", color: "#2764a8", points: validatedPoints },
      { label: "探索性外推值", color: "#8a6b35", points: exploratoryPoints },
      {
        label: "失效阈值",
        color: "#aa2b2b",
        points: thresholdPoints,
        continuous: true,
      },
    ],
  });
}
function drawDeepLearningChart(report) {
  const canvas = $("deepLearningChart");
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = canvas.height / ratio;
  ctx.clearRect(0, 0, width, height);
  const modelId = $("deepModelSelect")?.value || report.bestTestModel;
  const group = deepModelSeries(report, modelId);
  const testSeries = group.series;
  $("deepErrorTitle").textContent = `${group.modelName} / 五参数归一化绝对误差`;
  const pad = { left: 72, right: 24, top: 24, bottom: 54 };
  const seriesList = testSeries.map((series) => ({
    name: paramLabel(series.feature),
    points: series.points.map((point) => ({
      x: point.targetIndex,
      y: point.normalizedAbsErrorPercent,
    })),
  }));
  if (!seriesList.length || !seriesList.some((series) => series.points.length)) return;
  const scale = scaleForXY(seriesList, width, height, pad);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657174";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText(`${fmt(scale.max, 2)}%`, 30, pad.top + 4);
  ctx.fillText(`${fmt(scale.min, 2)}%`, 30, height - pad.bottom);
  ctx.fillText(String(Math.round(scale.minX)), pad.left, height - pad.bottom + 18);
  ctx.fillText(String(Math.round(scale.maxX)), width - pad.right - 38, height - pad.bottom + 18);
  drawAxisLabels(
    ctx,
    width,
    height,
    pad,
    "横轴：独立测试目标行号",
    "纵轴：归一化绝对误差 (%)",
  );
  testSeries.forEach((series, idx) => {
    const color = palette[idx % palette.length];
    const points = series.points.map((point) => ({
      x: point.targetIndex,
      y: point.normalizedAbsErrorPercent,
    }));
    drawLineXY(ctx, points, scale, color, 2, []);
    const shortLabels = {
      VoltageMaxRaw: "最大电压",
      VoltageFirstZeroTimeUs: "首次过零时间",
      VoltageReversePeakCoefficient: "反峰系数",
      VoltageMinAbsRaw: "最小电压",
      DischargePeriodSec: "放电周期",
    };
    const legendX = Math.max(pad.left + 8, width - 180);
    ctx.fillStyle = color;
    ctx.fillRect(legendX, pad.top + idx * 20 - 9, 14, 4);
    ctx.fillText(
      shortLabels[series.feature] || paramLabel(series.feature),
      legendX + 22,
      pad.top + idx * 20,
    );
  });
  registerChartHover(canvas, {
    pad,
    scale,
    xFormatter: (value) => `独立测试目标行：${Math.round(value)}`,
    series: testSeries.map((series, idx) => ({
      label: `${paramLabel(series.feature)}误差 (%)`,
      color: palette[idx % palette.length],
      points: series.points.map((point) => ({
        x: point.targetIndex,
        y: point.normalizedAbsErrorPercent,
      })),
    })),
  });
}
function downloadReport() {
  if (!state.latestEvaluation) return;
  const payload = {
    evaluation: state.latestEvaluation,
    validation: state.latestValidation,
    modelComparison: state.latestModelComparison,
    deepLearning: state.latestDeepLearning,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `pulse-capacitor-report-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}
$("dataset").addEventListener("change", () => {
  renderColumnChoices();
  if (state.latestEvaluation) {
    state.latestEvaluation = null;
    state.latestValidation = null;
    state.latestModelComparison = null;
  }
});
function activateWorkspacePage(pageId) {
  state.activePage = pageId;
  document.querySelectorAll(".workspace-tab").forEach((tab) => {
    const active = tab.dataset.page === pageId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".workspace-page").forEach((page) => {
    const active = page.id === pageId;
    page.classList.toggle("is-active", active);
    page.hidden = !active;
  });
  requestAnimationFrame(() => {
    if (pageId === "patentPage") {
      if (state.latestEvaluation) drawChart(state.latestEvaluation);
      if (state.latestValidation) drawValidationChart(state.latestValidation);
    } else if (pageId === "alternativePage" && state.latestModelComparison) {
      drawAlternativeChart(state.latestModelComparison);
      drawAlternativeValidationChart(state.latestModelComparison);
    } else if (pageId === "deepLearningPage" && state.latestDeepLearning) {
      renderDeepLife(state.latestDeepLearning);
      drawDeepPredictionChart(state.latestDeepLearning);
      drawDeepLearningChart(state.latestDeepLearning);
    }
  });
}
$("run").addEventListener("click", runEvaluation);
$("download").addEventListener("click", downloadReport);
document.querySelectorAll(".workspace-tab").forEach((tab) => {
  tab.addEventListener("click", () => activateWorkspacePage(tab.dataset.page));
});
[$("deepModelSelect"), $("deepFeatureSelect")].forEach((select) => {
  select.addEventListener("change", () => {
    if (!state.latestDeepLearning) return;
    drawDeepPredictionChart(state.latestDeepLearning);
    drawDeepLearningChart(state.latestDeepLearning);
  });
});
$("deepLifeFeatureSelect").addEventListener("change", () => {
  if (!state.latestDeepLearning) return;
  drawDeepLifeChart(state.latestDeepLearning);
});
$("deepLifeLimit").addEventListener("change", () => {
  if (!state.latestDeepLearning) return;
  renderDeepLife(state.latestDeepLearning);
});
document.querySelectorAll('input[name="predictionMode"], input[name="validationMode"]').forEach((input) => {
  input.addEventListener("change", syncModeControls);
});
window.addEventListener("resize", () => {
  if (state.activePage === "patentPage") {
    if (state.latestEvaluation) drawChart(state.latestEvaluation);
    if (state.latestValidation) drawValidationChart(state.latestValidation);
  } else if (state.activePage === "alternativePage" && state.latestModelComparison) {
    drawAlternativeChart(state.latestModelComparison);
    drawAlternativeValidationChart(state.latestModelComparison);
  } else if (state.activePage === "deepLearningPage" && state.latestDeepLearning) {
    drawDeepLifeChart(state.latestDeepLearning);
    drawDeepPredictionChart(state.latestDeepLearning);
    drawDeepLearningChart(state.latestDeepLearning);
  }
});
async function initialize() {
  await loadDatasets();
  await loadDeepLearningReport();
  await runEvaluation();
}
syncModeControls();
activateWorkspacePage("deepLearningPage");
initialize().catch((error) => alert(error.message));
