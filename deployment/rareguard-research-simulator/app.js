"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  data: null,
  trace: [],
  index: 0,
  timer: null,
  recent: [],
};

const POLICY_LABELS = {
  rareguard_agent: "RareGuard agent",
  always_light_nova: "Always-light NOVA-XIDS",
  always_heavy_rareguard: "Always-heavy RareGuard-XIDS",
};

const MODEL_LABELS = {
  "RareGuard-XIDS": "RareGuard-XIDS",
  "NOVA-XIDS": "NOVA-XIDS",
  "RareGuard-no-class-balance": "Ablation: no class balance",
  "RareGuard-no-contrastive": "Ablation: no contrastive",
  "RareGuard-no-direct-path": "Ablation: no direct path",
  "RareGuard-no-prototype": "Ablation: no prototype",
  RandomForest: "Random Forest",
  ExtraTrees: "Extra Trees",
  HistGradientBoosting: "Hist. Gradient Boosting",
};

const METRICS = {
  f1_macro: { label: "Macro F1", higher: true },
  rare_f1_macro: { label: "Rare-class F1", higher: true },
  accuracy: { label: "Accuracy", higher: true },
  ece_15: { label: "ECE", higher: false },
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function bytes(value) {
  const units = ["B", "KB", "MB", "GB"];
  let n = Number(value);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function humanize(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function currentDataset() { return $("datasetSelect").value; }
function currentSeed() { return Number($("seedSelect").value); }
function currentPolicy() { return $("policySelect").value; }
function currentPreset() { return $("presetSelect").value; }

function datasetInfo(dataset = currentDataset()) {
  return state.data.datasets.find((item) => item.id === dataset);
}

function selectionRow(dataset = currentDataset(), seed = currentSeed()) {
  return state.data.validation_selections.find(
    (row) => row.dataset === dataset && Number(row.seed) === Number(seed),
  );
}

function policyRow(policy = currentPolicy(), dataset = currentDataset(), seed = currentSeed()) {
  return state.data.policy_metrics.find(
    (row) => row.dataset === dataset && Number(row.seed) === Number(seed) && row.policy === policy,
  );
}

function activeThresholds() {
  const row = selectionRow();
  const base = {
    uncertainty: Number(row.uncertainty_threshold),
    rare: Number(row.rare_probability_threshold),
    drift: Number(row.drift_threshold),
  };
  if (currentPreset() === "sensitivity") {
    return {
      uncertainty: Math.max(0, base.uncertainty - 0.05),
      rare: Math.max(0, base.rare - 0.05),
      drift: Math.max(0, base.drift - 0.002),
    };
  }
  if (currentPreset() === "efficiency") {
    return {
      uncertainty: Math.min(1, base.uncertainty + 0.05),
      rare: Math.min(1, base.rare + 0.05),
      drift: Math.min(1, base.drift + 0.002),
    };
  }
  return base;
}

function evaluateEvent(event) {
  const policy = currentPolicy();
  if (policy === "always_light_nova") {
    return { wake: false, reasons: ["Policy fixed to light detector"], prediction: event.light_prediction };
  }
  if (policy === "always_heavy_rareguard") {
    return { wake: true, reasons: ["Policy fixed to heavy detector"], prediction: event.heavy_prediction };
  }
  if (currentPreset() === "validated") {
    const reasons = event.trigger_reason === "none"
      ? ["No validation-selected trigger crossed"]
      : String(event.trigger_reason).split(";").map(humanize);
    return {
      wake: Boolean(event.wake_heavy_detector),
      reasons,
      prediction: event.wake_heavy_detector ? event.heavy_prediction : event.light_prediction,
    };
  }
  const thresholds = activeThresholds();
  const reasons = [];
  if (Number(event.uncertainty) >= thresholds.uncertainty) reasons.push("Predictive uncertainty");
  if (Number(event.rare_probability) >= thresholds.rare) reasons.push("Rare attack signal");
  if (Number(event.drift_score) >= thresholds.drift) reasons.push("Traffic drift");
  const wake = reasons.length > 0;
  return {
    wake,
    reasons: wake ? reasons : ["No diagnostic threshold crossed"],
    prediction: wake ? event.heavy_prediction : event.light_prediction,
  };
}

function setSignal(prefix, value, threshold, active) {
  $(`${prefix}Value`).textContent = fmt(value, 4);
  $(`${prefix}Bar`).style.width = `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
  $(`${prefix}Marker`).style.left = `${Math.max(0, Math.min(100, Number(threshold) * 100))}%`;
  $(`${prefix}State`).textContent = active ? "Threshold crossed" : "Below threshold";
  $(`${prefix}State`).style.color = active ? "var(--amber)" : "var(--muted)";
}

function renderThresholds() {
  const thresholds = activeThresholds();
  $("uncertaintyThreshold").textContent = fmt(thresholds.uncertainty, 4);
  $("rareThreshold").textContent = fmt(thresholds.rare, 4);
  $("driftThreshold").textContent = fmt(thresholds.drift, 4);
  const preset = currentPreset();
  $("presetHelp").textContent = preset === "validated"
    ? "Uses the exact recorded gate decision selected on validation data."
    : preset === "sensitivity"
      ? "Diagnostic replay: lowers thresholds over the same recorded scores. Not a benchmark row."
      : "Diagnostic replay: raises thresholds over the same recorded scores. Not a benchmark row.";
}

function updatePipeline(event, decision) {
  const correct = String(decision.prediction) === String(event.true_class);
  $("sourceEventIndex").textContent = `Source index ${event.index}`;
  $("lightPrediction").textContent = `Prediction ${event.light_prediction}`;
  $("gateDecision").textContent = decision.wake ? "Escalate to heavy" : "Retain light result";
  $("heavyPrediction").textContent = decision.wake ? `Prediction ${event.heavy_prediction}` : "Bypassed for this event";
  $("outcomeText").textContent = correct ? "Correct classification" : "Misclassification";
  $("trueClass").textContent = event.true_class;
  $("gateNode").className = `pipeline-node gate ${decision.wake ? "active" : ""}`;
  $("heavyNode").className = `pipeline-node heavy ${decision.wake ? "active" : "bypassed"}`;
  $("outcomeNode").className = `pipeline-node outcome ${correct ? "success" : "failure"}`;
  $("explanationTitle").textContent = decision.wake
    ? `Heavy detector activated by ${decision.reasons.join(", ").toLowerCase()}.`
    : "Light-detector decision retained; escalation gate stayed closed.";
  $("explanationBody").textContent = currentPreset() === "validated"
    ? `Recorded policy reason: ${decision.reasons.join("; ")}. Final prediction ${decision.prediction}; true class ${event.true_class}.`
    : `Diagnostic threshold replay over saved scores: ${decision.reasons.join("; ")}. This outcome is not a reported experimental row.`;
}

function appendLog(event, decision) {
  const correct = String(decision.prediction) === String(event.true_class);
  state.recent.unshift({ event, decision, correct });
  state.recent = state.recent.slice(0, 7);
  $("eventLogBody").innerHTML = state.recent.map(({ event: item, decision: route, correct: ok }) => `
    <tr>
      <td>${escapeHtml(item.index)}</td>
      <td>${route.wake ? "Light → Heavy" : "Light only"}</td>
      <td>${escapeHtml(route.reasons.join("; "))}</td>
      <td>${escapeHtml(route.prediction)}</td>
      <td>${escapeHtml(item.true_class)}</td>
      <td><span class="result-pill ${ok ? "correct" : "incorrect"}">${ok ? "Correct" : "Incorrect"}</span></td>
    </tr>`).join("");
}

function renderEvent(addToLog = true) {
  if (!state.trace.length) return;
  const event = state.trace[state.index];
  const decision = evaluateEvent(event);
  const thresholds = activeThresholds();
  $("eventCounter").textContent = `Event ${state.index + 1} / ${state.trace.length}`;
  const validated = currentPreset() === "validated";
  setSignal("uncertainty", event.uncertainty, thresholds.uncertainty, validated ? event.trigger_uncertainty : Number(event.uncertainty) >= thresholds.uncertainty);
  setSignal("rare", event.rare_probability, thresholds.rare, validated ? event.trigger_rare_signal : Number(event.rare_probability) >= thresholds.rare);
  setSignal("drift", event.drift_score, thresholds.drift, validated ? event.trigger_drift : Number(event.drift_score) >= thresholds.drift);
  updatePipeline(event, decision);
  if (addToLog) appendLog(event, decision);
}

function stopReplay() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  $("playButton").textContent = "Start replay";
}

function stepReplay() {
  state.index = (state.index + 1) % state.trace.length;
  renderEvent(true);
}

function startReplay() {
  if (state.timer) { stopReplay(); return; }
  $("playButton").textContent = "Pause replay";
  state.timer = window.setInterval(stepReplay, Number($("speedSelect").value));
}

function computeWindowMetrics() {
  const evaluations = state.trace.map((event) => ({ event, decision: evaluateEvent(event) }));
  const heavy = evaluations.filter((item) => item.decision.wake).length;
  const correct = evaluations.filter((item) => String(item.decision.prediction) === String(item.event.true_class)).length;
  const ratio = evaluations.length ? heavy / evaluations.length : 0;
  $("windowAccuracy").textContent = pct(correct / evaluations.length, 1);
  $("windowHeavyRatio").textContent = pct(ratio, 1);
  $("windowAccuracyNote").textContent = currentPreset() === "validated" ? "Contiguous recorded window" : "Diagnostic replay window";
  $("windowHeavyNote").textContent = `${heavy} of ${evaluations.length} replay events`;

  const active = policyRow();
  let projected = active?.projected_inference_seconds_from_measured_runs;
  let note = "Reported projection from measured runs";
  if (currentPolicy() === "rareguard_agent" && currentPreset() !== "validated") {
    const light = policyRow("always_light_nova");
    const heavyRow = policyRow("always_heavy_rareguard");
    projected = Number(light.projected_inference_seconds_from_measured_runs)
      + ratio * (Number(heavyRow.projected_inference_seconds_from_measured_runs) - Number(light.projected_inference_seconds_from_measured_runs));
    note = "Diagnostic interpolation from measured endpoints";
  }
  $("projectedSeconds").textContent = `${fmt(projected, 2)} s`;
  $("projectedTimeNote").textContent = note;
}

function refreshReplay() {
  stopReplay();
  const dataset = currentDataset();
  const seed = String(currentSeed());
  state.trace = state.data.event_traces[dataset][seed];
  state.index = 0;
  state.recent = [];
  const info = datasetInfo();
  const traceMeta = state.data.event_trace_summary.trace_windows[`${dataset}:${seed}`];
  $("datasetRole").textContent = info.role === "external_confirmation" ? "External confirmation" : "Corrective development";
  $("datasetRole").className = `role-chip ${info.role === "external_confirmation" ? "confirmation" : ""}`;
  $("datasetRows").textContent = `${Number(traceMeta.source_events).toLocaleString()} recorded events`;
  $("presetSelect").disabled = currentPolicy() !== "rareguard_agent";
  if (currentPolicy() !== "rareguard_agent") $("presetSelect").value = "validated";
  renderThresholds();
  computeWindowMetrics();
  renderEvent(true);
}

function populateSelectors() {
  const options = state.data.datasets.map((item) => `<option value="${item.id}">${escapeHtml(item.label)}</option>`).join("");
  $("datasetSelect").innerHTML = options;
  $("evidenceDataset").innerHTML = options;
  $("seedSelect").innerHTML = state.data.seeds.map((seed) => `<option value="${seed}">${seed}</option>`).join("");
  $("datasetSelect").value = "ciciot2023";
  $("evidenceDataset").value = "ciciot2023";
}

function renderSummary() {
  const gates = Object.values(state.data.readiness.gates);
  const passed = gates.filter(Boolean).length;
  $("readinessStatus").textContent = state.data.readiness.status;
  $("gateCount").textContent = `${passed} of ${gates.length}`;
  $("gateMeter").style.width = `${(passed / gates.length) * 100}%`;
  $("datasetCount").textContent = state.data.datasets.length;
  $("decisionCount").textContent = Number(state.data.event_trace_summary.total_events).toLocaleString();
  $("heavyRatio").textContent = pct(state.data.event_trace_summary.overall_heavy_call_ratio, 1);
  $("generatedAt").textContent = `Bundle generated ${new Date(state.data.generated_at_utc).toLocaleString()}`;
}

function evidenceRows(dataset, metric) {
  const central = state.data.central_summary
    .filter((row) => row.dataset === dataset)
    .map((row) => ({ model: row.model, value: row[`${metric}_mean`], family: row.model.startsWith("RareGuard") ? "central" : "comparator" }));
  const baselines = state.data.baseline_summary
    .filter((row) => row.dataset === dataset)
    .map((row) => ({ model: row.model, value: row[metric], family: "baseline" }));
  return [...central, ...baselines].filter((row) => row.value !== null && row.value !== undefined);
}

function renderEvidence() {
  const dataset = $("evidenceDataset").value;
  const metric = $("metricSelect").value;
  const spec = METRICS[metric];
  const rows = evidenceRows(dataset, metric).sort((a, b) => spec.higher ? b.value - a.value : a.value - b.value);
  const max = Math.max(...rows.map((row) => Number(row.value)), 0.0001);
  const info = datasetInfo(dataset);
  $("evidenceRole").textContent = info.role === "external_confirmation" ? "External confirmation" : "Corrective development";
  $("evidenceRole").className = `role-chip ${info.role === "external_confirmation" ? "confirmation" : ""}`;
  $("chartTitle").textContent = `${info.label} · ${spec.label}`;
  $("chartDirection").textContent = spec.higher ? "Higher is better" : "Lower is better";
  $("modelBars").innerHTML = rows.map((row) => `
    <div class="model-row ${row.model === "RareGuard-XIDS" ? "rareguard" : row.family === "baseline" ? "baseline" : ""}">
      <span class="model-label" title="${escapeHtml(MODEL_LABELS[row.model] || row.model)}">${escapeHtml(MODEL_LABELS[row.model] || row.model)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(1, Number(row.value) / max * 100)}%"></div></div>
      <span class="bar-value">${fmt(row.value, 4)}</span>
    </div>`).join("");

  const rg = rows.find((row) => row.model === "RareGuard-XIDS");
  const comparators = rows.filter((row) => row.model !== "RareGuard-XIDS" && !row.model.startsWith("RareGuard-no-"));
  const best = [...comparators].sort((a, b) => spec.higher ? b.value - a.value : a.value - b.value)[0];
  const signed = Number(rg.value) - Number(best.value);
  const leads = spec.higher ? signed > 0 : signed < 0;
  const tie = Math.abs(signed) < 1e-8;
  $("comparisonHeadline").textContent = tie
    ? `RareGuard-XIDS is tied with ${MODEL_LABELS[best.model] || best.model} on this displayed metric.`
    : `RareGuard-XIDS ${leads ? "leads" : "trails"} the best non-ablation comparator here.`;
  $("comparisonNarrative").textContent = `${spec.label} is displayed for ${info.label} using saved seed-aggregated evidence. This local comparison does not override the final cross-dataset readiness gates.`;
  $("rareguardMetric").textContent = fmt(rg.value, 4);
  $("bestComparatorMetric").textContent = `${fmt(best.value, 4)} · ${MODEL_LABELS[best.model] || best.model}`;
  $("displayedDifference").textContent = `${signed >= 0 ? "+" : ""}${fmt(signed, 4)}`;
}

function findingHtml(title, subtitle, value, status = "neutral") {
  return `<div class="finding ${status}"><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(subtitle)}</small></div><span class="finding-value">${escapeHtml(value)}</span></div>`;
}

function renderResearchFindings() {
  const robustness = state.data.robustness_comparisons;
  const robustPass = robustness.filter((row) => row.passed).length;
  $("robustnessScore").textContent = `${robustPass}/${robustness.length} pass`;
  $("robustnessList").innerHTML = robustness.map((row) => findingHtml(
    datasetInfo(row.dataset).label,
    `${humanize(row.perturbation)} · strength ${row.strength}`,
    `${Number(row.mean_primary_composite_gain) >= 0 ? "+" : ""}${fmt(row.mean_primary_composite_gain, 4)}`,
    row.passed ? "pass" : "fail",
  )).join("");

  const ablations = state.data.ablation_paired_gains;
  const ablationWins = ablations.filter((row) => Number(row.mean_gain_full_minus_ablation) > 0).length;
  $("ablationScore").textContent = `${ablationWins}/${ablations.length} positive`;
  $("ablationList").innerHTML = ablations.map((row) => findingHtml(
    `${datasetInfo(row.dataset).label} · ${humanize(row.metric)}`,
    MODEL_LABELS[row.ablation] || row.ablation,
    `${Number(row.mean_gain_full_minus_ablation) >= 0 ? "+" : ""}${fmt(row.mean_gain_full_minus_ablation, 4)}`,
    Number(row.mean_gain_full_minus_ablation) > 0 ? "pass" : "fail",
  )).join("");

  const continual = state.data.continual_comparisons;
  const continualPass = continual.filter((row) => row.passed).length;
  $("continualScore").textContent = `${continualPass}/${continual.length} pass`;
  $("continualList").innerHTML = continual.map((row) => findingHtml(
    `${datasetInfo(row.dataset).label} · ${humanize(row.metric)}`,
    `Comparator: ${row.baseline}`,
    `${Number(row.difference) >= 0 ? "+" : ""}${fmt(row.difference, 4)}`,
    row.passed ? "pass" : "fail",
  )).join("");

  const federatedGroups = new Map();
  state.data.federated_summary.forEach((row) => {
    if (!federatedGroups.has(row.method)) federatedGroups.set(row.method, []);
    federatedGroups.get(row.method).push(row);
  });
  const fedRows = [...federatedGroups.entries()].map(([method, rows]) => ({
    method,
    f1: rows.reduce((sum, row) => sum + Number(row.f1_macro), 0) / rows.length,
    communication: rows.reduce((sum, row) => sum + Number(row.communication_bytes), 0) / rows.length,
  })).sort((a, b) => b.f1 - a.f1);
  $("federatedList").innerHTML = fedRows.map((row) => findingHtml(
    row.method,
    `Mean communication ${bytes(row.communication)}`,
    `F1 ${fmt(row.f1, 4)}`,
    "neutral",
  )).join("");
}

function renderReproducibility() {
  const gates = Object.entries(state.data.readiness.gates);
  const passed = gates.filter(([, value]) => value).length;
  $("gateSummary").textContent = `${passed}/${gates.length} pass`;
  $("gateList").innerHTML = gates.map(([name, value]) => `
    <div class="gate-item ${value ? "pass" : "fail"}"><span class="gate-icon">${value ? "✓" : "!"}</span><span>${escapeHtml(humanize(name))}</span></div>`).join("");
  $("disclosureList").innerHTML = state.data.disclosures.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  $("manifestCount").textContent = `${state.data.source_manifest.length} files`;
  $("manifestBody").innerHTML = state.data.source_manifest.map((item) => `
    <tr><td>${escapeHtml(item.path)}</td><td>${bytes(item.bytes)}</td><td title="${item.sha256}">${item.sha256}</td></tr>`).join("");
}

function bindEvents() {
  ["datasetSelect", "seedSelect", "policySelect", "presetSelect"].forEach((id) => $(id).addEventListener("change", refreshReplay));
  $("speedSelect").addEventListener("change", () => { if (state.timer) { stopReplay(); startReplay(); } });
  $("playButton").addEventListener("click", startReplay);
  $("stepButton").addEventListener("click", stepReplay);
  $("resetButton").addEventListener("click", refreshReplay);
  $("evidenceDataset").addEventListener("change", renderEvidence);
  $("metricSelect").addEventListener("change", renderEvidence);
  $("printButton").addEventListener("click", () => window.print());
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopReplay(); });
}

async function init() {
  try {
    const response = await fetch("assets/data/evidence.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Evidence request failed: HTTP ${response.status}`);
    state.data = await response.json();
    populateSelectors();
    renderSummary();
    renderResearchFindings();
    renderReproducibility();
    bindEvents();
    refreshReplay();
    renderEvidence();
    $("loadingOverlay").classList.add("hidden");
    const captureTarget = new URLSearchParams(window.location.search).get("capture");
    if (captureTarget) {
      $("loadingOverlay").style.display = "none";
      document.documentElement.style.scrollBehavior = "auto";
      const target = document.getElementById(captureTarget);
      if (target) target.scrollIntoView({ block: "start" });
    }
  } catch (error) {
    $("loadingOverlay").innerHTML = `<p><strong>Evidence bundle could not be loaded.</strong><br>${escapeHtml(error.message)}<br>Serve this folder through HTTP rather than opening index.html directly.</p>`;
  }
}

init();
