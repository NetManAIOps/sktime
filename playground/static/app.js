const state = {
  catalog: null,
  result: null,
  view: { start: 0, length: 0 },
  viewKey: null,
};

const el = (id) => document.getElementById(id);

async function init() {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }
  window.scrollTo(0, 0);
  setStatus("Loading catalog");
  const response = await fetch("/api/catalog");
  state.catalog = await response.json();
  renderCatalog();
  window.scrollTo(0, 0);
  setTimeout(() => window.scrollTo(0, 0), 0);
  bindEvents();
  setStatus("Ready");
}

function bindEvents() {
  ["taskSelect", "sourceSelect", "datasetSelect", "preprocessorSelect", "algorithmSelect"].forEach((id) => {
    el(id).addEventListener("change", () => {
      if (id === "taskSelect" || id === "sourceSelect" || id === "algorithmSelect") {
        refreshOptions();
      }
      if (id === "taskSelect") {
        renderPreprocessorOptions();
      }
      renderParams();
      renderPipeline();
    });
  });
  el("runButton").addEventListener("click", runExperiment);
  el("detectorSubtypeSelect").addEventListener("change", () => {
    refreshOptions();
    renderParams();
    renderPipeline();
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  });
  el("viewStart").addEventListener("input", () => {
    state.view.start = Number(el("viewStart").value);
    renderChart();
  });
  el("viewLength").addEventListener("input", () => {
    const total = state.result?.series?.points?.length || 1;
    state.view.length = Math.max(1, Math.min(Number(el("viewLength").value) || 1, total));
    state.view.start = Math.min(state.view.start, total - state.view.length);
    renderChart();
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.result) renderChart();
    }, 120);
  });
}

function renderCatalog() {
  const taskSelect = el("taskSelect");
  taskSelect.innerHTML = state.catalog.tasks
    .map((task) => `<option value="${task.id}">${task.label}</option>`)
    .join("");
  renderDependencies();
  renderPreprocessorOptions();
  refreshOptions();
  renderParams();
  renderPipeline();
  renderEmptyState();
}

function renderDependencies() {
  const deps = state.catalog.dependencies || {};
  const critical = ["pandas", "scikit-base", "scikit-learn", "scipy"];
  const optional = ["huggingface-hub", "pyarrow"];
  const lines = [...critical, ...optional].map((name) => {
    const item = deps[name] || { available: false };
    const cls = item.available ? "dep-ok" : "dep-missing";
    const suffix = item.available ? "available" : `missing`;
    return `<div class="dep-line ${cls}">${name}: ${suffix}</div>`;
  });
  const hf = state.catalog.hf || {};
  const hfConfigs = Array.isArray(hf.configs) ? hf.configs.length : 0;
  lines.push(`<div class="dep-line ${hf.available ? "dep-ok" : "dep-missing"}">HF metadata: ${hf.available ? "available" : "offline"}</div>`);
  lines.push(`<div class="dep-line ${hfConfigs ? "dep-ok" : "dep-missing"}">HF datasets: ${hfConfigs} configs</div>`);
  el("dependencies").innerHTML = lines.join("");
}

function renderPreprocessorOptions() {
  const task = el("taskSelect")?.value;
  const previous = el("preprocessorSelect").value;
  const preprocessors = (state.catalog.preprocessors || []).filter((item) => {
    if (!item.enabled) return false;
    const compat = item.compatible_tasks;
    if (!compat || compat.includes("all") || !task) return true;
    return compat.includes(task);
  });
  el("preprocessorSelect").innerHTML = preprocessors
    .map((item) => `<option value="${item.id}">${item.name}</option>`)
    .join("");
  if ([...el("preprocessorSelect").options].some((option) => option.value === previous)) {
    el("preprocessorSelect").value = previous;
  } else {
    el("preprocessorSelect").value = "none";
  }
}

function refreshOptions() {
  const task = el("taskSelect").value;
  const source = el("sourceSelect").value;
  const previousAlgorithm = el("algorithmSelect").value;
  const algorithms = state.catalog.algorithms.filter((algorithm) => algorithm.task === task);
  const subtypeWrap = el("subtypeWrap");
  if (subtypeWrap) subtypeWrap.hidden = task !== "anomaly_detection";
  let visibleAlgorithms = algorithms;
  if (task === "anomaly_detection") {
    const subtype = el("detectorSubtypeSelect")?.value || "all";
    if (subtype !== "all") {
      visibleAlgorithms = algorithms.filter((algorithm) => (algorithm.subtype || "other") === subtype);
    }
  }
  const enabledAlgorithms = visibleAlgorithms.filter((algorithm) => algorithm.enabled);
  el("algorithmSelect").innerHTML = [
    ...enabledAlgorithms.map((algorithm) => `<option value="${algorithm.id}">${algorithm.name}</option>`),
    ...visibleAlgorithms
      .filter((algorithm) => !algorithm.enabled)
      .slice(0, 20)
      .map((algorithm) => `<option value="${algorithm.id}" disabled>${algorithm.name} - ${algorithm.disabled_reason}</option>`),
  ].join("");
  if ([...el("algorithmSelect").options].some((option) => option.value === previousAlgorithm)) {
    el("algorithmSelect").value = previousAlgorithm;
  }

  const algorithmId = el("algorithmSelect").value;
  const compatibleIds = new Set(
    state.catalog.compatibility
      .filter((row) => row.task === task && row.algorithm_id === algorithmId)
      .map((row) => row.dataset_id)
  );
  let datasets = state.catalog.datasets.filter((dataset) => dataset.task === task && compatibleIds.has(dataset.id));
  if (source !== "all") {
    datasets = datasets.filter((dataset) => dataset.source === source);
  }
  el("datasetSelect").innerHTML = datasets
    .map((dataset) => `<option value="${dataset.id}">${dataset.name}${dataset.source === "huggingface" ? " (online)" : ""}</option>`)
    .join("");
  el("compatHint").textContent = `${datasets.length} compatible datasets for this task and algorithm.`;
  renderPipeline();
}

function renderParams() {
  const algorithm = currentAlgorithm();
  const preprocessor = currentPreprocessor();
  const algorithmParams = algorithm?.params || {};
  const preprocessorParams = preprocessor?.params || {};
  const preprocessorInputs = Object.entries(preprocessorParams).map(([key, value]) => {
    const label = `pre: ${key.replaceAll("_", " ")}`;
    return `<label>${label}<input data-preprocessor-param="${key}" type="number" step="any" value="${value}"></label>`;
  });
  const algorithmInputs = Object.entries(algorithmParams).map(([key, value]) => {
    const label = key.replaceAll("_", " ");
    return `<label>${label}<input data-param="${key}" type="number" step="any" value="${value}"></label>`;
  });
  el("params").innerHTML = [...preprocessorInputs, ...algorithmInputs].join("");
}

function renderPipeline(result = null) {
  const task = result?.task || el("taskSelect").value;
  const dataset = result?.dataset || currentDataset();
  const algorithm = result?.algorithm || currentAlgorithm();
  const selectedPreprocessor = result?.preprocessor || currentPreprocessor();
  const preprocessor = preprocessorFor(task, algorithm, dataset, selectedPreprocessor);
  const steps = [
    {
      stage: "Dataset",
      title: dataset?.name || "Select dataset",
      detail: dataset ? `${dataset.source}${dataset.online ? " · online" : ""}` : "No dataset selected",
    },
    {
      stage: "Preprocessor",
      title: preprocessor.name,
      detail: preprocessor.detail,
    },
    {
      stage: "Algorithm",
      title: algorithm?.name || "Select algorithm",
      detail: algorithm?.module || "No algorithm selected",
    },
    {
      stage: "Evaluation",
      title: evaluationTitle(result),
      detail: evaluationDetail(result),
    },
  ];
  el("pipeline").innerHTML = steps
    .map(
      (step) => `<div class="pipeline-step">
        <span class="stage">${escapeHtml(step.stage)}</span>
        <strong>${escapeHtml(step.title)}</strong>
        <small>${escapeHtml(step.detail)}</small>
      </div>`
    )
    .join("");
}

function preprocessorFor(task, algorithm, dataset, selectedPreprocessor = null) {
  if (selectedPreprocessor && selectedPreprocessor.id !== "none") {
    return {
      name: selectedPreprocessor.name,
      detail: selectedPreprocessor.module || "Dynamic sktime transformer preprocessor.",
    };
  }
  if (task === "anomaly_detection" && algorithm?.id === "threshold-detector") {
    return {
      name: "Rolling median + z-score",
      detail: "Detrend strong trend/seasonality before ThresholdDetector.",
    };
  }
  if (task === "classification" && algorithm?.name === "SummaryClassifier") {
    return {
      name: "SummaryTransformer",
      detail: "sktime feature-based classifier preprocessing summary features.",
    };
  }
  if (task === "forecasting") {
    if (dataset?.source === "huggingface") {
      return {
        name: "dropna + RangeIndex",
        detail: "Clean values and normalize HF timestamps to a reproducible index.",
      };
    }
    return {
      name: "dropna",
      detail: "Minimal forecasting preprocessing before fit/predict.",
    };
  }
  return {
    name: "Identity / none",
    detail: "No task-specific preprocessor is applied in this run.",
  };
}

function evaluationTitle(result) {
  if (!result) return "Pending";
  if (result.status !== "ok") return "Blocked";
  return "Metrics + table + log";
}

function evaluationDetail(result) {
  if (!result) return "Run an experiment to populate this stage.";
  if (result.status !== "ok") return result.error || "Runtime returned a blocked result.";
  const metrics = Object.entries(result.metrics || {})
    .slice(0, 3)
    .map(([name, value]) => `${name}=${formatValue(value)}`)
    .join(" · ");
  return metrics || result.summary || "Experiment finished.";
}

async function runExperiment() {
  const params = {};
  document.querySelectorAll("[data-param]").forEach((input) => {
    params[input.dataset.param] = Number(input.value);
  });
  const preprocessorParams = {};
  document.querySelectorAll("[data-preprocessor-param]").forEach((input) => {
    preprocessorParams[input.dataset.preprocessorParam] = Number(input.value);
  });
  const spec = {
    task: el("taskSelect").value,
    dataset_id: el("datasetSelect").value,
    algorithm_id: el("algorithmSelect").value,
    preprocessor_id: el("preprocessorSelect").value,
    params,
    preprocessor_params: preprocessorParams,
  };
  showProgress("Running");
  el("runButton").disabled = true;
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    const result = await response.json();
    state.result = result;
    if (result.status !== "ok") {
      renderError(result.error || "Unknown error");
      renderPipeline(result);
      setStatus("Blocked");
    } else {
      renderResult(result);
      setStatus("Complete");
    }
  } catch (error) {
    renderError(String(error));
    setStatus("Error");
  } finally {
    hideProgress();
    el("runButton").disabled = false;
  }
}

function renderResult(result) {
  el("runMeta").textContent = `${result.dataset.name} / ${result.algorithm.name} / ${result.duration_ms} ms`;
  renderMetrics(result.metrics);
  renderTable(result.tables);
  state.view = { start: 0, length: 0 };
  state.viewKey = null;
  renderChart();
  renderPipeline(result);
  el("codeBlock").textContent = result.code || "";
  el("reportBlock").textContent = result.report || "";
  el("logBlock").textContent = (result.log || []).join("\n");
  el("scriptLink").href = `/api/export/script?run_id=${result.run_id}`;
  el("reportLink").href = `/api/export/report?run_id=${result.run_id}`;
}

function renderMetrics(metrics) {
  el("metrics").innerHTML = Object.entries(metrics || {})
    .map(([name, value]) => `<div class="metric"><span>${name}</span><strong>${formatValue(value)}</strong></div>`)
    .join("");
}

function renderTable(tables) {
  const wrap = el("tableWrap");
  if (!tables) {
    wrap.innerHTML = "";
    return;
  }
  if (tables.forecast) {
    wrap.innerHTML = htmlTable(tables.forecast);
  } else if (tables.predictions) {
    const cm = tables.confusion_matrix;
    wrap.innerHTML = htmlTable(tables.predictions) + confusionMatrix(cm);
  } else if (tables.detections) {
    wrap.innerHTML = htmlTable(tables.detections);
  } else {
    wrap.innerHTML = "<p>No table output.</p>";
  }
}

function htmlTable(rows) {
  if (!rows || rows.length === 0) return "<p>No rows.</p>";
  const columns = Object.keys(rows[0]);
  return `<table><thead><tr>${columns.map((col) => `<th>${col}</th>`).join("")}</tr></thead><tbody>${rows
    .map((row) => `<tr>${columns.map((col) => `<td>${formatValue(row[col])}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function confusionMatrix(cm) {
  if (!cm) return "";
  const header = `<tr><th>actual \\ predicted</th>${cm.labels.map((label) => `<th>${label}</th>`).join("")}</tr>`;
  const rows = cm.matrix
    .map((row, i) => `<tr><th>${cm.labels[i]}</th>${row.map((value) => `<td>${value}</td>`).join("")}</tr>`)
    .join("");
  return `<h3 style="margin-top:18px">Confusion Matrix</h3><table><thead>${header}</thead><tbody>${rows}</tbody></table>`;
}

function setupCanvas() {
  const canvas = el("chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(320, canvas.clientWidth || canvas.parentElement?.clientWidth || 960);
  const cssH = Math.max(220, canvas.clientHeight || 360);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cssW, h: cssH };
}

function renderChart() {
  const { ctx, w, h } = setupCanvas();
  ctx.fillStyle = "#fffaf5";
  ctx.fillRect(0, 0, w, h);
  const result = state.result;
  const series = result?.series;
  if (!series || !series.points || series.points.length === 0) {
    drawLabel(ctx, "Run an experiment to render a chart.", 30, 50);
    toggleChartControls(false);
    return;
  }
  if (series.kind === "classification") {
    toggleChartControls(false);
    drawConfusionHeatmap(ctx, result.tables?.confusion_matrix, w, h);
    return;
  }
  toggleChartControls(true);
  ensureView(series);
  const { start, length } = state.view;
  const slice = series.points.slice(start, start + length);
  if (series.kind === "forecast") {
    drawForecastWindow(ctx, slice, w, h);
  } else {
    drawAnomalyWindow(ctx, slice, w, h);
  }
  syncChartControls(series);
}

function ensureView(series) {
  const key = state.result?.run_id;
  if (state.viewKey === key && state.view.length > 0) return;
  const meta = series.meta || {};
  const total = series.points.length;
  let start = 0;
  let length = total;
  if (series.kind === "forecast") {
    const cw = Number(state.result.spec?.params?.context_window) || meta.context_window || 36;
    const hz = meta.horizon || 1;
    length = Math.min(total, cw + hz);
    const testStart = meta.test_start ?? total;
    start = Math.max(0, Math.min(testStart - cw, total - length));
  } else {
    length = Math.min(total, 600);
    const firstHit = series.points.findIndex((p) => p.predicted_anomaly);
    if (firstHit >= 0) {
      start = Math.max(0, Math.min(firstHit - Math.floor(length / 3), total - length));
    } else {
      start = Math.max(0, total - length);
    }
  }
  state.view = { start, length };
  state.viewKey = key;
}

function syncChartControls(series) {
  const total = series.points.length;
  const startInput = el("viewStart");
  const lenInput = el("viewLength");
  startInput.min = 0;
  startInput.max = Math.max(0, total - state.view.length);
  startInput.value = state.view.start;
  lenInput.min = 1;
  lenInput.max = total;
  lenInput.value = state.view.length;
  const end = Math.min(state.view.start + state.view.length, total);
  el("viewHint").textContent = `showing ${state.view.start}\u2013${end} of ${total}`;
}

function toggleChartControls(show) {
  const controls = el("chartControls");
  if (controls) controls.hidden = !show;
}

function drawForecastWindow(ctx, points, w, h) {
  const values = [];
  points.forEach((p) => {
    if (typeof p.actual === "number") values.push(p.actual);
    if (typeof p.prediction === "number") values.push(p.prediction);
  });
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 38;
  const firstTest = points.findIndex((p) => p.split === "test");
  if (firstTest >= 0) {
    const x0 = pad + (firstTest / Math.max(points.length - 1, 1)) * (w - pad * 2);
    ctx.fillStyle = "rgba(255, 111, 0, 0.07)";
    ctx.fillRect(x0, pad, w - pad - x0, h - pad * 2);
  }
  drawAxes(ctx, w, h, pad);
  drawLine(ctx, points, "actual", "#5d4037", min, max, w, h, pad);
  drawLine(ctx, points, "prediction", "#ff6f00", min, max, w, h, pad);
  drawLegend(ctx, [["actual", "#5d4037"], ["prediction", "#ff6f00"], ["test region", "rgba(255,111,0,0.35)"]]);
}

function drawAnomalyWindow(ctx, points, w, h) {
  const values = points.map((p) => p.value).filter((v) => typeof v === "number");
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 38;
  drawAxes(ctx, w, h, pad);
  drawLine(ctx, points, "value", "#5d4037", min, max, w, h, pad);
  drawAnomalies(ctx, points, min, max, w, h, pad);
  drawLegend(ctx, [["signal", "#5d4037"], ["predicted anomaly", "#ff6f00"], ["ground truth", "#2e7d32"]]);
}

function drawConfusionHeatmap(ctx, cm, w, h) {
  if (!cm || !cm.labels || cm.labels.length === 0) {
    drawLabel(ctx, "No confusion matrix available.", 30, 50);
    return;
  }
  const { labels, matrix } = cm;
  const n = labels.length;
  let maxCount = 1;
  matrix.forEach((row) => row.forEach((v) => { if (v > maxCount) maxCount = v; }));
  const leftPad = 72;
  const topPad = 40;
  const bottomPad = 66;
  const rightPad = 20;
  const cellW = (w - leftPad - rightPad) / n;
  const cellH = (h - topPad - bottomPad) / n;
  const showCounts = n <= 12;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const t = v / maxCount;
      const x = leftPad + j * cellW;
      const y = topPad + i * cellH;
      if (i === j) {
        const g = Math.round(232 - t * 120);
        ctx.fillStyle = `rgb(${g}, 245, ${g})`;
      } else if (v > 0) {
        ctx.fillStyle = `rgb(255, ${Math.round(205 - t * 130)}, ${Math.round(175 - t * 120)})`;
      } else {
        ctx.fillStyle = "#fff7ef";
      }
      ctx.fillRect(x, y, cellW - 1, cellH - 1);
      if (showCounts) {
        ctx.fillStyle = t > 0.55 ? "#ffffff" : "#5d4037";
        ctx.font = `${Math.max(10, Math.min(14, cellW / 3))}px system-ui`;
        ctx.textAlign = "center";
        ctx.fillText(String(v), x + cellW / 2, y + cellH / 2 + 4);
        ctx.textAlign = "left";
      }
    }
  }
  ctx.fillStyle = "#5d4037";
  ctx.font = "11px system-ui";
  labels.forEach((label, i) => {
    ctx.save();
    ctx.translate(leftPad - 8, topPad + i * cellH + cellH / 2 + 4);
    ctx.textAlign = "right";
    ctx.fillText(truncateLabel(String(label), 10), 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(leftPad + i * cellW + cellW / 2, h - bottomPad + 16);
    ctx.rotate(-Math.PI / 6);
    ctx.textAlign = "right";
    ctx.fillText(truncateLabel(String(label), 10), 0, 0);
    ctx.restore();
  });
  ctx.textAlign = "left";
  drawLabel(ctx, "rows = actual \u00b7 cols = predicted \u00b7 darker = more \u00b7 green diagonal = correct", leftPad, 24);
}

function truncateLabel(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}\u2026` : text;
}

function drawLine(ctx, points, key, color, min, max, width, height, pad) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  let started = false;
  points.forEach((point, i) => {
    if (typeof point[key] !== "number") return;
    const x = pad + (i / Math.max(points.length - 1, 1)) * (width - pad * 2);
    const y = scale(point[key], min, max, height, pad);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function drawAnomalies(ctx, points, min, max, width, height, pad) {
  points.forEach((point, i) => {
    const x = pad + (i / Math.max(points.length - 1, 1)) * (width - pad * 2);
    const y = scale(point.value, min, max, height, pad);
    if (point.actual_anomaly) {
      ctx.fillStyle = "#2e7d32";
      ctx.fillRect(x - 2, y - 9, 4, 18);
    }
    if (point.predicted_anomaly) {
      ctx.fillStyle = "#ff6f00";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

function drawClassification(ctx, points, width, height) {
  const labels = [...new Set(points.flatMap((p) => [p.actual, p.prediction]))];
  const pad = 42;
  drawAxes(ctx, width, height, pad);
  points.forEach((point, i) => {
    const x = pad + (i / Math.max(points.length - 1, 1)) * (width - pad * 2);
    const actualY = pad + labels.indexOf(point.actual) * 28;
    const predY = actualY + 12;
    ctx.fillStyle = point.actual === point.prediction ? "#2e7d32" : "#c62828";
    ctx.fillRect(x, predY, 5, 14);
  });
  drawLabel(ctx, "Each bar is a test prediction; red marks mismatch.", pad, height - 18);
}

function drawAxes(ctx, width, height, pad) {
  ctx.strokeStyle = "#ead8c8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();
}

function drawLegend(ctx, items) {
  items.forEach(([label, color], index) => {
    const x = 46 + index * 150;
    ctx.fillStyle = color;
    ctx.fillRect(x, 18, 14, 8);
    drawLabel(ctx, label, x + 20, 26);
  });
}

function drawLabel(ctx, text, x, y) {
  ctx.fillStyle = "#8a6f5d";
  ctx.font = "13px system-ui";
  ctx.fillText(text, x, y);
}

function scale(value, min, max, height, pad) {
  if (max === min) return height / 2;
  return height - pad - ((value - min) / (max - min)) * (height - pad * 2);
}

function renderEmptyState() {
  state.result = null;
  renderChart();
  el("metrics").innerHTML = "";
  el("tableWrap").innerHTML = "<p>Select a task, dataset, and algorithm, then run an experiment.</p>";
  el("codeBlock").textContent = "Generated reproduction code appears after a successful run.";
  el("reportBlock").textContent = "Experiment report appears after a successful run.";
  el("logBlock").textContent = "Execution log appears after a successful run.";
}

function renderError(message) {
  el("metrics").innerHTML = `<div class="metric"><span>Status</span><strong>Blocked</strong></div>`;
  el("tableWrap").innerHTML = `<p>${escapeHtml(message)}</p>`;
  el("logBlock").textContent = message;
  showTab("log");
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  el("codeBlock").hidden = name !== "code";
  el("reportBlock").hidden = name !== "report";
  el("logBlock").hidden = name !== "log";
}

function currentAlgorithm() {
  return state.catalog.algorithms.find((algorithm) => algorithm.id === el("algorithmSelect").value);
}

function currentDataset() {
  return state.catalog.datasets.find((dataset) => dataset.id === el("datasetSelect").value);
}

function currentPreprocessor() {
  return (state.catalog.preprocessors || []).find((item) => item.id === el("preprocessorSelect").value);
}

function setStatus(text) {
  el("statusPill").textContent = text;
}

function showProgress(text) {
  const bar = el("progressBar");
  if (bar) bar.hidden = false;
  if (text) setStatus(text);
}

function hideProgress() {
  const bar = el("progressBar");
  if (bar) bar.hidden = true;
}

function formatValue(value) {
  if (typeof value === "number") {
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(4);
    return value.toPrecision(4);
  }
  return String(value);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));
}

init().catch((error) => {
  setStatus("Error");
  renderError(String(error));
});
