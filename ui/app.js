const form = document.getElementById("predictForm");
const predictButton = document.getElementById("predictButton");
const presetRow = document.getElementById("presetRow");
const benchmarkGrid = document.getElementById("benchmarkGrid");
const featureList = document.getElementById("featureList");
const galleryGrid = document.getElementById("galleryGrid");

const pm25Value = document.getElementById("pm25Value");
const riskBadge = document.getElementById("riskBadge");
const confidenceRange = document.getElementById("confidenceRange");
const confidenceLabel = document.getElementById("confidenceLabel");
const narrativeText = document.getElementById("narrativeText");
const driverList = document.getElementById("driverList");
const factorBars = document.getElementById("factorBars");
const riskFill = document.getElementById("riskFill");

let presets = [];

function formatRangeValue(input) {
  const target = document.getElementById(`${input.id}Value`);
  if (!target) return;

  let suffix = "";
  if (input.id === "temperature" || input.id === "dew_point") suffix = "°C";
  if (input.id === "humidity") suffix = "%";
  if (input.id === "pressure") suffix = " hPa";
  if (input.id === "wind_speed") suffix = " m/s";
  if (input.id === "wind_direction") suffix = "°";
  if (input.id === "boundary_layer_height") suffix = " m";

  target.textContent = `${input.value}${suffix}`;
}

function formToPayload() {
  const data = new FormData(form);
  const payload = {};

  for (const [key, value] of data.entries()) {
    if (value === "") continue;
    if (key === "timestamp") {
      payload[key] = value;
      continue;
    }
    payload[key] = Number(value);
  }

  return payload;
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
      <div class="benchmark-metric"><span>R²</span><strong>${row.r2.toFixed(4)}</strong></div>
      <div class="benchmark-metric"><span>RMSE</span><strong>${row.rmse.toFixed(2)}</strong></div>
      <div class="benchmark-metric"><span>MAE</span><strong>${row.mae.toFixed(2)}</strong></div>
      <div class="benchmark-metric"><span>SMAPE</span><strong>${row.smape.toFixed(2)}%</strong></div>
    `;
    benchmarkGrid.appendChild(card);
  });
}

function renderFeatures(items) {
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
      </div>
    `;
    galleryGrid.appendChild(card);
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

function renderPrediction(result) {
  pm25Value.textContent = result.pm25.toFixed(1);
  riskBadge.textContent = result.band.label;
  riskBadge.style.background = `${result.band.color}22`;
  riskBadge.style.color = result.band.color;
  confidenceRange.textContent = `${result.confidence.low.toFixed(1)} to ${result.confidence.high.toFixed(1)} ug/m3`;
  confidenceLabel.textContent = result.confidence.label;
  narrativeText.textContent = result.narrative;
  riskFill.style.width = `${Math.min(100, result.band.severity)}%`;
  renderDrivers(result.drivers);
  renderFactorBars(result.factor_scores);
}

async function runPrediction() {
  predictButton.disabled = true;
  predictButton.textContent = "Running...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formToPayload()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed");
    renderPrediction(result);
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
  presets = summary.project.presets || [];
  renderPresets();
  renderBenchmarks(summary.project.benchmarks || []);
  renderFeatures(summary.project.top_features || []);
  renderGallery(summary.project.figures || []);

  if (presets[0]) {
    setFormValues(presets[0].payload);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runPrediction();
  });

  predictButton.addEventListener("click", runPrediction);
  runPrediction();
}

bootstrap();
