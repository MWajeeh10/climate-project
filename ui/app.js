const form = document.getElementById("predictForm");
const predictButton = document.getElementById("predictButton");
const presetRow = document.getElementById("presetRow");
const benchmarkGrid = document.getElementById("benchmarkGrid");
const dailyBenchmarkGrid = document.getElementById("dailyBenchmarkGrid");
const dailySuiteNote = document.getElementById("dailySuiteNote");
const dailyCasesBody = document.getElementById("dailyCasesBody");
const accuracyTableBody = document.getElementById("accuracyTableBody");
const featureList = document.getElementById("featureList");
const galleryGrid = document.getElementById("galleryGrid");
const researchGrid = document.getElementById("researchGrid");
const glossaryGrid = document.getElementById("glossaryGrid");
const modelBlendGrid = document.getElementById("modelBlendGrid");
const publicGuidanceList = document.getElementById("publicGuidanceList");
const modelNote = document.getElementById("modelNote");
const heroModelNote = document.getElementById("heroModelNote");
const calcGrid = document.getElementById("calcGrid");
const modeSwitch = document.getElementById("modeSwitch");
const modeNote = document.getElementById("modeNote");
const guidedSummary = document.getElementById("guidedSummary");
const resetTypicalButton = document.getElementById("resetTypicalButton");
const modelSelector = document.getElementById("model_key");

const pm25Value = document.getElementById("pm25Value");
const riskBadge = document.getElementById("riskBadge");
const confidenceRange = document.getElementById("confidenceRange");
const confidenceLabel = document.getElementById("confidenceLabel");
const narrativeText = document.getElementById("narrativeText");
const driverList = document.getElementById("driverList");
const factorBars = document.getElementById("factorBars");
const riskFill = document.getElementById("riskFill");

let presets = [];
let summaryProject = null;
let currentMode = "guided";
let forecastChartInstance = null;

function formatRangeValue(input) {
  const target = document.getElementById(`${input.id}Value`);
  if (!target) return;

  let suffix = "";
  if (input.id === "temperature" || input.id === "dew_point") suffix = " C";
  if (input.id === "humidity") suffix = "%";
  if (input.id === "pressure") suffix = " hPa";
  if (input.id === "wind_speed") suffix = " m/s";
  if (input.id === "wind_direction") suffix = " deg";
  if (input.id === "boundary_layer_height") suffix = " m";

  target.textContent = `${input.value}${suffix}`;
}

function formToPayload() {
  const data = new FormData(form);
  const payload = {};

  for (const [key, value] of data.entries()) {
    if (value === "") continue;
    if (key === "timestamp" || key === "model_key") {
      payload[key] = value;
      continue;
    }
    if (currentMode === "guided" && key !== "traffic_factor") {
      continue;
    }
    payload[key] = Number(value);
  }

  return payload;
}

function isDailyModel() {
  return String(modelSelector.value || "").startsWith("daily_");
}

function setFormValues(payload) {
  Object.entries(payload).forEach(([key, value]) => {
    const field = form.elements.namedItem(key);
    if (!field) return;
    field.value = value;
    if (field.type === "range") formatRangeValue(field);
  });
}

function renderPresets() {
  presetRow.innerHTML = "";
  presets.forEach((preset, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-chip";
    if (index === 0) button.classList.add("active");
    button.innerHTML = `<strong>${preset.title}</strong><span>${preset.description}</span>`;
    button.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach((chip) => chip.classList.remove("active"));
      button.classList.add("active");
      setFormValues(preset.payload);
      runPrediction();
    });
    presetRow.appendChild(button);
  });
}

function renderBenchmarks(rows) {
  benchmarkGrid.innerHTML = "";
  rows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "benchmark-card";
    card.innerHTML = `
      <h3>${row.model}</h3>
      <div class="benchmark-metric"><span>R2</span><strong>${row.r2.toFixed(4)}</strong></div>
      <div class="benchmark-metric"><span>RMSE</span><strong>${row.rmse.toFixed(2)}</strong></div>
      <div class="benchmark-metric"><span>MAE</span><strong>${row.mae.toFixed(2)}</strong></div>
      <div class="benchmark-metric"><span>SMAPE</span><strong>${row.smape.toFixed(2)}%</strong></div>
    `;
    benchmarkGrid.appendChild(card);
  });
}

function renderDailyBenchmarks(rows) {
  dailyBenchmarkGrid.innerHTML = "";
  rows.forEach((row) => {
    const card = document.createElement("article");
    card.className = "benchmark-card";
    const keyLabel = row.model_key.replaceAll("_", " ");
    card.innerHTML = `
      <h3>${keyLabel}</h3>
      <div class="benchmark-metric"><span>R2</span><strong>${row.r2.toFixed(4)}</strong></div>
      <div class="benchmark-metric"><span>RMSE</span><strong>${row.rmse.toFixed(2)}</strong></div>
      <div class="benchmark-metric"><span>MAE</span><strong>${row.mae.toFixed(2)}</strong></div>
    `;
    dailyBenchmarkGrid.appendChild(card);
  });
}

function renderDailyCases(rows) {
  dailyCasesBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.scenario}</td>
      <td>${row.date}</td>
      <td>${Number(row.actual_pm25).toFixed(1)}</td>
      <td>${Number(row.daily_random_forest).toFixed(1)}</td>
      <td>${Number(row.daily_ridge).toFixed(1)}</td>
      <td>${Number(row.daily_xgboost).toFixed(1)}</td>
      <td>${Number(row.daily_lightgbm).toFixed(1)}</td>
      <td>${Number(row.daily_extra_trees).toFixed(1)}</td>
      <td>${Number(row.daily_weighted_ensemble).toFixed(1)}</td>
    `;
    dailyCasesBody.appendChild(tr);
  });
}

function renderModelSelector(models) {
  modelSelector.innerHTML = "";
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = model.label;
    modelSelector.appendChild(option);
  });
  if (models.find((item) => item.key === "daily_random_forest")) {
    modelSelector.value = "daily_random_forest";
  }
}

function renderAccuracyMatrix(rows) {
  accuracyTableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const highlight = row.model === "Weighted Avg" ? " class=\"best-row\"" : "";
    tr.innerHTML = `
      <tr${highlight}>
        <td>${row.model}</td>
        <td>${row.r2.toFixed(4)}</td>
        <td>${row.rmse.toFixed(2)}</td>
        <td>${row.mae.toFixed(2)}</td>
        <td>${row.smape.toFixed(2)}</td>
        <td>${row.episode_r2.toFixed(4)}</td>
        <td>${row.hazard_detection.toFixed(1)}</td>
        <td>${row.false_alarm.toFixed(1)}</td>
      </tr>
    `;
    accuracyTableBody.appendChild(tr.firstElementChild);
  });
}

function renderFeatures(items) {
  if (!featureList) return;
  featureList.innerHTML = "";
  items.forEach((item, index) => {
    const chip = document.createElement("article");
    chip.className = "feature-chip";
    chip.innerHTML = `
      <p class="eyebrow">Signal ${index + 1}</p>
      <h3>${item.feature.replaceAll("_", " ")}</h3>
      <p>Combined importance score: ${item.combined.toFixed(1)}</p>
    `;
    featureList.appendChild(chip);
  });
}

function renderGallery(items) {
  galleryGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "gallery-card";
    card.innerHTML = `
      <img src="${item.path}" alt="${item.title}">
      <div class="gallery-copy">
        <h3>${item.title}</h3>
        <p>${item.caption}</p>
        <p><strong>How to read it:</strong> ${item.reading}</p>
        <p><strong>Why it matters:</strong> ${item.impact}</p>
      </div>
    `;
    galleryGrid.appendChild(card);
  });
}

function renderResearch(items) {
  if (!researchGrid) return;
  researchGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "research-card";
    card.innerHTML = `
      <p class="eyebrow">Project strength</p>
      <h3>${item.title}</h3>
      <p>${item.body}</p>
    `;
    researchGrid.appendChild(card);
  });
}

function renderCalculationSteps(items) {
  if (!calcGrid) return;
  calcGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "calc-card";
    card.innerHTML = `
      <h3>${item.step}</h3>
      <p>${item.detail}</p>
    `;
    calcGrid.appendChild(card);
  });
}

function renderGlossary(items) {
  glossaryGrid.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "glossary-card";
    card.innerHTML = `
      <h3>${item.term}</h3>
      <p>${item.meaning}</p>
    `;
    glossaryGrid.appendChild(card);
  });
}

function renderDrivers(drivers) {
  driverList.innerHTML = "";
  if (!drivers || !drivers.length) {
    driverList.innerHTML = "<li>No strong driver dominated this scenario.</li>";
    return;
  }

  drivers.forEach((driver) => {
    const li = document.createElement("li");
    li.textContent = driver;
    driverList.appendChild(li);
  });
}

function renderPublicGuidance(items) {
  publicGuidanceList.innerHTML = "";
  if (!items || !items.length) {
    publicGuidanceList.innerHTML = "<li>Run a prediction to generate plain-language public guidance.</li>";
    return;
  }

  items.forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = entry;
    publicGuidanceList.appendChild(li);
  });
}

function renderFactorBars(items) {
  factorBars.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "factor-row";
    row.innerHTML = `
      <div class="factor-label"><span>${item.label}</span><strong>${item.value}</strong></div>
      <div class="factor-track"><div class="factor-bar" style="width:${item.value}%"></div></div>
    `;
    factorBars.appendChild(row);
  });
}

function renderModelBlend(items, bestSavedModel) {
  modelBlendGrid.innerHTML = "";
  const maxValue = Math.max(...items.map((item) => item.pm25), 1);
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "blend-row";
    const width = Math.max(8, (item.pm25 / maxValue) * 100);
    let badge = "";
    if (item.name === "Weighted Avg") badge = "Headline prediction";
    else if (item.name === bestSavedModel) badge = "Best saved benchmark";

    row.innerHTML = `
      <div class="blend-head">
        <span>${item.name}</span>
        <strong>${item.pm25.toFixed(1)} ug/m3</strong>
      </div>
      ${badge ? `<p class="blend-tag">${badge}</p>` : ""}
      <div class="factor-track"><div class="factor-bar blend-bar" style="width:${width}%"></div></div>
    `;
    modelBlendGrid.appendChild(row);
  });
}

function renderPrediction(result) {
  pm25Value.textContent = result.pm25.toFixed(1);
  riskBadge.textContent = result.band.label;
  riskBadge.style.background = `${result.band.color}22`;
  riskBadge.style.color = result.band.color;
  confidenceRange.textContent = `${result.confidence.low.toFixed(1)} to ${result.confidence.high.toFixed(1)} ug/m3`;
  confidenceLabel.textContent = result.confidence.label;
  narrativeText.textContent = result.narrative;
  riskFill.style.width = `${Math.min(100, result.band.severity)}%`;
  modelNote.textContent = result.model_note || "";
  if (result.input_mode === "Guided defaults" && result.auto_defaults_preview) {
    const auto = result.auto_defaults_preview;
    if (isDailyModel()) {
      guidedSummary.textContent = `Guided mode used typical daily Lahore conditions for this month: about ${auto.temperature} C, ${auto.humidity}% humidity, ${auto.wind_speed} m/s wind, previous-day PM2.5 near ${auto.pm25_lag_1d} ug/m3, and previous-week PM2.5 near ${auto.pm25_lag_7d} ug/m3.`;
    } else {
      guidedSummary.textContent = `Guided mode used typical historical Lahore conditions for this month/hour: about ${auto.temperature} C, ${auto.humidity}% humidity, ${auto.wind_speed} m/s wind, boundary layer near ${auto.boundary_layer_height} m, and recent PM2.5 around ${auto.pm25_lag1h} ug/m3.`;
    }
  }
  renderDrivers(result.drivers);
  renderPublicGuidance(result.public_guidance || []);
  renderFactorBars(result.factor_scores);
  renderModelBlend(result.live_models || [], result.best_saved_model);
}

function renderForecastChart(rangeData) {
  const ctx = document.getElementById('forecastChart');
  if (!ctx) return;

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  const labels = rangeData.map(d => {
    const dt = new Date(d.timestamp);
    return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  });
  
  const dataPoints = rangeData.map(d => d.pm25);
  
  // Choose colors based on the highest risk band in the forecast
  const maxPm25 = Math.max(...dataPoints);
  let color = '#4ade80'; // Good
  if (maxPm25 > 35) color = '#facc15'; // Moderate
  if (maxPm25 > 150) color = '#f87171'; // Unhealthy
  if (maxPm25 > 300) color = '#a78bfa'; // Hazardous
  
  forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Predicted PM2.5 (ug/m3)',
        data: dataPoints,
        borderColor: color,
        backgroundColor: color + '33', // 20% opacity
        borderWidth: 3,
        pointBackgroundColor: color,
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return ' ' + context.parsed.y.toFixed(1) + ' ug/m3';
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: {
            color: '#334155',
            drawBorder: false
          },
          ticks: {
            color: '#94a3b8'
          }
        },
        x: {
          grid: {
            display: false
          },
          ticks: {
            color: '#94a3b8',
            maxTicksLimit: 8
          }
        }
      }
    }
  });
}

async function applyTypicalDefaults() {
  const timestamp = form.elements.namedItem("timestamp").value;
  const trafficFactor = Number(form.elements.namedItem("traffic_factor").value || 1.0);
  const response = await fetch("/api/autofill", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ timestamp, traffic_factor: trafficFactor, model_key: modelSelector.value }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load typical conditions");
  setFormValues(data.defaults);
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-chip").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  document.body.classList.toggle("guided-mode", mode === "guided");
  modeNote.textContent = mode === "guided"
    ? (isDailyModel()
        ? "Guided Forecast mode uses the cleaner observed daily dataset and fills weather plus lag inputs from typical monthly Lahore conditions."
        : "Guided Forecast mode hides the scientific controls and uses historical Lahore averages for the selected month and hour. This is best for non-technical users.")
    : (isDailyModel()
        ? "Manual Tuning mode lets you compare the observed daily models directly using temperature, humidity, wind speed, and previous-day / previous-week PM2.5."
        : "Manual Tuning mode exposes all weather, airflow, and pollution-memory controls so you can demonstrate the science behind the model.");
}

async function runPrediction() {
  predictButton.disabled = true;
  predictButton.textContent = "Running...";

  try {
    const payload = formToPayload();

    // 1. Run single-step prediction
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");
    renderPrediction(result);

    // 2. Fetch multi-step forecast range for the chart (skip for daily models as they are meant for 24h spans, not hourly steps)
    if (!isDailyModel()) {
      const rangePayload = { ...payload, steps: 48 };
      const rangeResponse = await fetch("/api/predict_range", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rangePayload),
      });
      if (rangeResponse.ok) {
        const rangeResult = await rangeResponse.json();
        renderForecastChart(rangeResult.steps);
      }
    } else {
      // Clear chart if we are using daily models
      if (forecastChartInstance) forecastChartInstance.destroy();
    }

  } catch (error) {
    narrativeText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Run Live Prediction";
  }
}

async function bootstrap() {
  document.querySelectorAll('input[type="range"]').forEach((input) => {
    formatRangeValue(input);
    input.addEventListener("input", () => formatRangeValue(input));
  });

  const response = await fetch("/api/summary");
  const summary = await response.json();
  summaryProject = summary.project;
  presets = summaryProject.presets || [];
  heroModelNote.textContent = summaryProject.model_note || "";
  modeNote.textContent = summaryProject.guided_mode_note || "";
  dailySuiteNote.textContent = summaryProject.daily_note || "";

  renderModelSelector(summaryProject.daily_models || []);
  renderPresets();
  renderBenchmarks(summaryProject.benchmarks || []);
  renderDailyBenchmarks(summaryProject.daily_model_metrics || []);
  renderDailyCases(summaryProject.daily_cases || []);
  renderAccuracyMatrix(summaryProject.benchmarks || []);
  renderFeatures(summaryProject.top_features || []);
  renderGallery(summaryProject.figures || []);
  renderResearch(summaryProject.research_highlights || []);
  renderGlossary(summaryProject.glossary || []);
  renderCalculationSteps(summaryProject.calculation_steps || []);

  if (presets[0]) {
    setFormValues(presets[0].payload);
  }
  setMode("guided");
  await applyTypicalDefaults();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runPrediction();
  });

  modeSwitch.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button) return;
    setMode(button.dataset.mode);
    if (currentMode === "guided") {
      await applyTypicalDefaults();
      await runPrediction();
    }
  });

  resetTypicalButton.addEventListener("click", async () => {
    await applyTypicalDefaults();
    if (currentMode === "guided") {
      await runPrediction();
    }
  });

  form.elements.namedItem("timestamp").addEventListener("change", async () => {
    if (currentMode === "guided") {
      await applyTypicalDefaults();
      await runPrediction();
    }
  });

  form.elements.namedItem("traffic_factor").addEventListener("change", async () => {
    if (currentMode === "guided") {
      await applyTypicalDefaults();
      await runPrediction();
    }
  });

  modelSelector.addEventListener("change", async () => {
    setMode(currentMode);
    await applyTypicalDefaults();
    await runPrediction();
  });

  predictButton.addEventListener("click", runPrediction);
  runPrediction();
}

bootstrap();
