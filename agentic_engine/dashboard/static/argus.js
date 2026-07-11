/**
 * ARGUS Command Dashboard — JavaScript
 * Handles SSE streaming, alert rendering, report generation,
 * threat gauge drawing, and analytics charts.
 */

"use strict";

// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────

const STATE = {
  alerts:        [],
  selectedAlert: null,
  filter:        "ALL",
  sse:           null,
  statsInterval: null,
  toastTimeout:  null,
};

const SEVERITY_COLORS = {
  CRITICAL:  "#FF1744",
  HIGH:      "#FF6D00",
  MODERATE:  "#FFD600",
  LOW:       "#00E676",
  NEGLIGIBLE:"#78909C",
};

const SEVERITY_ICONS = {
  CRITICAL:  "🔴",
  HIGH:      "🟠",
  MODERATE:  "🟡",
  LOW:       "🟢",
  NEGLIGIBLE:"⚪",
};

// ─────────────────────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const DOM = {
  clock:         $("header-clock"),
  statusDot:     $("status-dot"),
  statusLabel:   $("status-label"),
  alertList:     $("alert-list"),
  emptyState:    $("empty-state"),
  reportEmpty:   $("report-empty"),
  reportContent: $("report-content"),
  reportBadge:   $("report-badge"),
  gaugeCanvas:   $("threat-gauge"),
  gaugeScore:    $("gauge-score"),

  // Stats
  valTotal:      $("val-total"),
  valCritical:   $("val-critical"),
  valHigh:       $("val-high"),
  valModerate:   $("val-moderate"),
  valSubjects:   $("val-subjects"),
  valSubscribers:$("val-subscribers"),
  valNlp:        $("val-nlp"),

  // Report fields
  rptId:         $("report-id"),
  rptSeverity:   $("report-severity-badge"),
  rptSubject:    $("rpt-subject"),
  rptCamera:     $("rpt-camera"),
  rptTimestamp:  $("rpt-timestamp"),
  rptCoords:     $("rpt-coords"),
  rptAnomalyType:$("rpt-anomaly-type"),
  rptGaitClass:  $("rpt-gait-class"),
  barGaitConf:   $("bar-gait-conf"),
  lblGaitConf:   $("lbl-gait-conf"),
  barAnomaly:    $("bar-anomaly"),
  lblAnomaly:    $("lbl-anomaly"),
  barStress:     $("bar-stress"),
  lblStress:     $("lbl-stress"),
  behavioralDesc:$("behavioral-desc"),
  rptDeviation:  $("rpt-deviation"),
  barGnn:        $("bar-gnn"),
  lblGnn:        $("lbl-gnn"),
  barHotspot:    $("bar-hotspot"),
  lblHotspot:    $("lbl-hotspot"),
  proxBadge:     $("prox-badge"),
  barDigital:    $("bar-digital"),
  lblDigital:    $("lbl-digital"),
  rptMultiplier: $("rpt-multiplier"),
  rptRiskCat:    $("rpt-risk-cat"),
  flaggedPhrases:$("flagged-phrases"),
  synVision:     $("syn-vision"),
  synMultiplier: $("syn-multiplier"),
  synFinal:      $("syn-final"),
  recBox:        $("recommendation-box"),
  btnNarrative:  $("btn-narrative"),
  narrativeText: $("narrative-text"),
  anomalyChart:  $("anomaly-chart"),
  subjectsList:  $("subjects-list"),
  toast:         $("alert-toast"),
  toastTitle:    $("toast-title"),
  toastBody:     $("toast-body"),
  toastClose:    $("toast-close"),
};

// ─────────────────────────────────────────────────────────────
// CLOCK
// ─────────────────────────────────────────────────────────────

function updateClock() {
  const now = new Date();
  const ts = now.toUTCString().replace("GMT", "UTC");
  DOM.clock.textContent = ts.slice(5, 25);
}
setInterval(updateClock, 1000);
updateClock();

// ─────────────────────────────────────────────────────────────
// SSE — REAL-TIME ALERT STREAM
// ─────────────────────────────────────────────────────────────

function connectSSE() {
  if (STATE.sse) STATE.sse.close();

  DOM.statusLabel.textContent = "CONNECTING...";
  DOM.statusDot.className = "status-dot";

  const es = new EventSource("/api/stream");

  es.addEventListener("connected", () => {
    DOM.statusDot.className = "status-dot online";
    DOM.statusLabel.textContent = "ARGUS ONLINE";
    $("pstat-dash").textContent = "LIVE";
    $("pstat-sensor").textContent = "ONLINE";
  });

  es.addEventListener("alert", (e) => {
    const alert = JSON.parse(e.data);
    STATE.alerts.unshift(alert);
    renderAlertCard(alert, true);
    updateStats();
    showToast(alert);
    updateAnalytics();
    if (STATE.selectedAlert && STATE.selectedAlert.alert_id === alert.alert_id) {
      renderReport(alert);
    }
  });

  es.addEventListener("heartbeat", () => {
    // Connection alive
  });

  es.onerror = () => {
    DOM.statusDot.className = "status-dot offline";
    DOM.statusLabel.textContent = "RECONNECTING...";
    setTimeout(connectSSE, 5000);
  };

  STATE.sse = es;
}

// ─────────────────────────────────────────────────────────────
// LOAD EXISTING ALERTS ON PAGE LOAD
// ─────────────────────────────────────────────────────────────

async function loadExistingAlerts() {
  try {
    const resp = await fetch("/api/alerts?per_page=50");
    const data = await resp.json();
    STATE.alerts = data.alerts || [];
    STATE.alerts.forEach(a => renderAlertCard(a, false));
    updateAnalytics();
  } catch(e) {
    console.warn("Could not load existing alerts:", e);
  }
}

async function loadStats() {
  try {
    const resp = await fetch("/api/stats");
    const data = await resp.json();

    DOM.valTotal.textContent     = data.total_stored || 0;
    DOM.valCritical.textContent  = (data.by_severity || {}).CRITICAL || 0;
    DOM.valHigh.textContent      = (data.by_severity || {}).HIGH || 0;
    DOM.valModerate.textContent  = (data.by_severity || {}).MODERATE || 0;
    DOM.valSubjects.textContent  = data.active_subjects || 0;
    DOM.valNlp.textContent       = data.nlp_enriched || 0;
    DOM.valSubscribers.textContent = (data.listener_stats || {}).total_received || 0;

    // Top subjects
    renderSubjectsList(data.top_threat_subjects || []);
  } catch(e) {}
}

function updateStats() {
  loadStats();
}

// ─────────────────────────────────────────────────────────────
// ALERT CARDS
// ─────────────────────────────────────────────────────────────

function renderAlertCard(alert, prepend = false) {
  const severity = alert.final_severity || alert.severity || "NEGLIGIBLE";
  const score = alert.final_threat_score ?? alert.composite_threat_index ?? 0;
  const type = alert.gait_class || alert.anomaly_type || "UNKNOWN";
  const ts = formatTimestamp(alert.timestamp);

  // Remove empty state
  if (DOM.emptyState) DOM.emptyState.style.display = "none";

  const el = document.createElement("div");
  el.className = `alert-card sev-${severity}`;
  el.dataset.alertId = alert.alert_id;
  el.setAttribute("role", "listitem");
  el.setAttribute("tabindex", "0");
  el.setAttribute("aria-label", `${severity} alert: ${type}`);

  const visionPct = Math.round((alert.composite_threat_index || 0) * 100);
  const routePct  = Math.round((alert.gnn_route_score || 0) * 100);
  const digitalPct= Math.round((alert.digital_risk_score || 0) * 100);

  el.innerHTML = `
    <div class="alert-card-header">
      <span class="alert-severity-pill pill-${severity}">${SEVERITY_ICONS[severity] || ''} ${severity}</span>
      <span class="alert-score">${(score * 100).toFixed(1)}%</span>
    </div>
    <div class="alert-type">${type.replace(/_/g, " ")}</div>
    <div class="alert-meta-row">
      <span class="alert-subject">${alert.subject_id || "—"}</span>
      <span class="alert-time">${ts}</span>
    </div>
    <div class="alert-mini-bars">
      <div class="mini-bar mb-vision"  style="width:${visionPct}%" title="Vision: ${visionPct}%"></div>
      <div class="mini-bar mb-route"   style="width:${routePct}%"  title="Route: ${routePct}%"></div>
      <div class="mini-bar mb-digital" style="width:${digitalPct}%" title="Digital: ${digitalPct}%"></div>
    </div>
  `;

  el.addEventListener("click", () => selectAlert(alert, el));
  el.addEventListener("keypress", (e) => {
    if (e.key === "Enter") selectAlert(alert, el);
  });

  if (prepend && DOM.alertList.firstChild) {
    DOM.alertList.insertBefore(el, DOM.alertList.firstChild);
  } else {
    DOM.alertList.appendChild(el);
  }
}

function applyFilter() {
  const cards = DOM.alertList.querySelectorAll(".alert-card");
  cards.forEach(card => {
    if (STATE.filter === "ALL") {
      card.style.display = "";
    } else {
      const sev = card.className.match(/sev-(\w+)/)?.[1];
      card.style.display = sev === STATE.filter ? "" : "none";
    }
  });
}

// ─────────────────────────────────────────────────────────────
// REPORT RENDERING
// ─────────────────────────────────────────────────────────────

function selectAlert(alert, cardEl) {
  // Deselect previous
  document.querySelectorAll(".alert-card.selected").forEach(c => c.classList.remove("selected"));
  if (cardEl) cardEl.classList.add("selected");

  STATE.selectedAlert = alert;
  renderReport(alert);
}

function renderReport(alert) {
  const severity = alert.final_severity || alert.severity || "NEGLIGIBLE";
  const finalScore = alert.final_threat_score ?? alert.composite_threat_index ?? 0;
  const color = SEVERITY_COLORS[severity] || "#78909C";

  // Show content, hide empty
  DOM.reportEmpty.style.display = "none";
  DOM.reportContent.style.display = "";

  // Badge
  DOM.reportBadge.textContent = `${SEVERITY_ICONS[severity]} ${severity} — ACTIVE`;
  DOM.reportBadge.style.color = color;
  DOM.reportBadge.style.borderColor = color + "60";

  // Hero
  DOM.rptId.textContent        = alert.alert_id || "—";
  DOM.rptSeverity.textContent  = `${SEVERITY_ICONS[severity] || ''} ${severity}`;
  DOM.rptSeverity.style.background = color + "22";
  DOM.rptSeverity.style.color      = color;
  DOM.rptSeverity.style.border     = `1px solid ${color}55`;

  DOM.rptSubject.textContent   = alert.subject_id || "—";
  DOM.rptCamera.textContent    = alert.camera_id  || "—";
  DOM.rptTimestamp.textContent = formatTimestamp(alert.timestamp);
  const coords = alert.location_coords || [0,0];
  DOM.rptCoords.textContent    = `${coords[0]?.toFixed(4)}°N, ${Math.abs(coords[1]?.toFixed(4))}°W`;

  // Gauge
  drawThreatGauge(finalScore, color);
  DOM.gaugeScore.textContent   = `${(finalScore * 100).toFixed(1)}%`;

  // Section 1 — Physical
  DOM.rptAnomalyType.textContent = (alert.anomaly_type || "—").replace(/_/g, " ");
  DOM.rptGaitClass.textContent   = (alert.gait_class || "—").replace(/_/g, " ");
  setBar(DOM.barGaitConf, DOM.lblGaitConf, alert.gait_confidence);
  setBar(DOM.barAnomaly,  DOM.lblAnomaly,  alert.anomaly_score);
  setBar(DOM.barStress,   DOM.lblStress,   alert.face_stress_index);
  DOM.behavioralDesc.textContent = getBehavioralDesc(alert.gait_class);

  // Section 2 — Spatiotemporal
  DOM.rptDeviation.textContent  = `${(alert.route_deviation_sigma || 0).toFixed(2)}σ`;
  setBar(DOM.barGnn,     DOM.lblGnn,     alert.gnn_route_score);
  setBar(DOM.barHotspot, DOM.lblHotspot, alert.hotspot_risk);

  if (alert.proximity_alert) {
    DOM.proxBadge.textContent  = "⚠ YES — ACTIVE";
    DOM.proxBadge.className    = "proximity-badge active";
  } else {
    DOM.proxBadge.textContent  = "NO";
    DOM.proxBadge.className    = "proximity-badge inactive";
  }

  // Section 3 — OSINT
  setBar(DOM.barDigital, DOM.lblDigital, alert.digital_risk_score);
  DOM.rptMultiplier.textContent = `${(alert.risk_multiplier || 1.0).toFixed(1)}x`;
  DOM.rptRiskCat.textContent    = (alert.risk_category || "NONE").replace(/_/g, " ");

  // Flagged phrases
  DOM.flaggedPhrases.innerHTML = "";
  const phrases = alert.flagged_phrases || [];
  if (phrases.length === 0) {
    const msg = document.createElement("span");
    msg.className = "no-flags-msg";
    msg.textContent = "✓ No predatory language detected in digital footprint.";
    DOM.flaggedPhrases.appendChild(msg);
  } else {
    phrases.slice(0, 8).forEach((p, i) => {
      const chip = document.createElement("span");
      chip.className = "flagged-chip";
      chip.textContent = `"${p}"`;
      chip.style.animationDelay = `${i * 0.06}s`;
      DOM.flaggedPhrases.appendChild(chip);
    });
  }

  // Section 4 — Synthesis
  DOM.synVision.textContent     = `${((alert.composite_threat_index || 0)*100).toFixed(1)}%`;
  DOM.synMultiplier.textContent = `${(alert.risk_multiplier || 1.0).toFixed(1)}x`;
  DOM.synFinal.textContent      = `${(finalScore * 100).toFixed(1)}%`;

  // Recommendation
  const rec = alert.report_narrative
    ? extractRecommendation(alert.report_narrative)
    : alert.structured_report?.recommendation || "Awaiting synthesis...";
  DOM.recBox.textContent = rec;
  DOM.recBox.className   = `recommendation-box rec-${severity}`;

  // Narrative
  if (alert.report_narrative) {
    DOM.narrativeText.textContent = alert.report_narrative;
    DOM.btnNarrative.style.display = "";
  } else {
    DOM.btnNarrative.style.display = "none";
  }
}

function setBar(barEl, labelEl, value) {
  const pct = Math.round((value || 0) * 100);
  barEl.style.width   = `${pct}%`;
  labelEl.textContent = `${pct}%`;
}

function getBehavioralDesc(gaitClass) {
  const descs = {
    PREDATORY_GAIT:     "Subject exhibits slow, deliberate ambulation with directional persistence toward a target, consistent with stalking locomotion patterns.",
    TAILING:            "Subject has maintained persistent spatial proximity to another individual over multiple frames, consistent with deliberate following behavior.",
    AGGRESSIVE_POSTURE: "Subject displays elevated limb extension, forward lean, and reduced bilateral symmetry in posture, consistent with pre-assault positioning.",
    LOITERING:          "Subject has remained stationary or performed repetitive low-radius movement within a defined area for an extended period without apparent purpose.",
    RAPID_APPROACH:     "Subject performed an accelerated, direct approach trajectory toward another individual or sensitive location.",
    CONCEALMENT:        "Subject is exhibiting body orientation and movement designed to minimize visual profile and avoid surveillance detection.",
    NORMAL:             "No significant anomaly detected in subject's gait or behavioral pattern.",
  };
  return descs[gaitClass] || "Behavioral pattern under analysis.";
}

function extractRecommendation(narrative) {
  const match = narrative.match(/RECOMMENDATION:\s*([\s\S]+?)(?:\n━|$)/);
  return match ? match[1].trim() : "See full narrative for recommendation.";
}

// ─────────────────────────────────────────────────────────────
// THREAT GAUGE (Canvas)
// ─────────────────────────────────────────────────────────────

function drawThreatGauge(score, color) {
  const canvas = DOM.gaugeCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const R  = 66;
  const startAngle = Math.PI * 0.75;
  const endAngle   = Math.PI * 2.25;
  const filled     = startAngle + (endAngle - startAngle) * score;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background track
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, endAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth   = 10;
  ctx.lineCap     = "round";
  ctx.stroke();

  // Score arc
  if (score > 0) {
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, "#00F5FF");
    gradient.addColorStop(1, color);
    ctx.beginPath();
    ctx.arc(cx, cy, R, startAngle, filled);
    ctx.strokeStyle = gradient;
    ctx.lineWidth   = 10;
    ctx.lineCap     = "round";
    ctx.stroke();

    // Glow effect
    ctx.beginPath();
    ctx.arc(cx, cy, R, startAngle, filled);
    ctx.strokeStyle = color + "55";
    ctx.lineWidth   = 18;
    ctx.lineCap     = "round";
    ctx.stroke();
  }

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const angle = startAngle + (endAngle - startAngle) * (i / 10);
    const innerR = R - 14;
    const outerR = R - 8;
    ctx.beginPath();
    ctx.moveTo(cx + innerR * Math.cos(angle), cy + innerR * Math.sin(angle));
    ctx.lineTo(cx + outerR * Math.cos(angle), cy + outerR * Math.sin(angle));
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth   = i % 5 === 0 ? 2 : 1;
    ctx.stroke();
  }
}

// ─────────────────────────────────────────────────────────────
// TOAST NOTIFICATION
// ─────────────────────────────────────────────────────────────

function showToast(alert) {
  const severity = alert.final_severity || alert.severity || "HIGH";
  const score    = alert.final_threat_score ?? alert.composite_threat_index ?? 0;
  const type     = (alert.gait_class || "ANOMALY").replace(/_/g, " ");
  const color    = SEVERITY_COLORS[severity] || "#FF6D00";

  DOM.toast.style.borderColor  = color;
  DOM.toast.style.boxShadow    = `0 0 20px ${color}66, 0 8px 32px rgba(0,0,0,0.6)`;
  DOM.toastTitle.style.color   = color;
  DOM.toastTitle.textContent   = `${SEVERITY_ICONS[severity] || '⚠'} ${severity} ALERT`;
  DOM.toastBody.textContent    = `${type} | Score: ${(score*100).toFixed(1)}% | ${alert.subject_id}`;

  DOM.toast.classList.add("show");

  clearTimeout(STATE.toastTimeout);
  STATE.toastTimeout = setTimeout(() => DOM.toast.classList.remove("show"), 6000);
}

DOM.toastClose?.addEventListener("click", () => {
  DOM.toast.classList.remove("show");
  clearTimeout(STATE.toastTimeout);
});

// ─────────────────────────────────────────────────────────────
// ANALYTICS CHARTS
// ─────────────────────────────────────────────────────────────

function updateAnalytics() {
  const anomalyCount = {};
  STATE.alerts.forEach(a => {
    const k = (a.gait_class || "UNKNOWN").replace(/_/g," ");
    anomalyCount[k] = (anomalyCount[k] || 0) + 1;
  });

  const maxCount = Math.max(...Object.values(anomalyCount), 1);
  const sorted   = Object.entries(anomalyCount).sort((a,b) => b[1]-a[1]).slice(0, 7);

  if (sorted.length === 0) {
    DOM.anomalyChart.innerHTML = '<div class="chart-empty">No data yet</div>';
    return;
  }

  DOM.anomalyChart.innerHTML = "";
  sorted.forEach(([label, count]) => {
    const pct = Math.round((count / maxCount) * 100);
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
      <span class="chart-label" title="${label}">${label}</span>
      <div class="chart-bar-wrap">
        <div class="chart-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="chart-count">${count}</span>
    `;
    DOM.anomalyChart.appendChild(row);
  });
}

function renderSubjectsList(subjects) {
  if (!subjects || subjects.length === 0) {
    DOM.subjectsList.innerHTML = '<div class="chart-empty">No subjects tracked</div>';
    return;
  }
  DOM.subjectsList.innerHTML = "";
  subjects.forEach(s => {
    const row = document.createElement("div");
    row.className = "subject-row";
    row.innerHTML = `
      <span class="subj-id">${s.subject_id}</span>
      <span class="subj-score" style="color:${getScoreColor(s.score)}">${(s.score*100).toFixed(1)}%</span>
    `;
    DOM.subjectsList.appendChild(row);
  });
}

function getScoreColor(score) {
  if (score >= 0.92) return SEVERITY_COLORS.CRITICAL;
  if (score >= 0.85) return SEVERITY_COLORS.HIGH;
  if (score >= 0.65) return SEVERITY_COLORS.MODERATE;
  return SEVERITY_COLORS.LOW;
}

// ─────────────────────────────────────────────────────────────
// FILTERS
// ─────────────────────────────────────────────────────────────

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    STATE.filter = btn.dataset.severity;
    applyFilter();
  });
});

// ─────────────────────────────────────────────────────────────
// SIMULATE BUTTON
// ─────────────────────────────────────────────────────────────

async function injectTestAlert() {
  try {
    const resp = await fetch("/api/inject-test", { method: "POST" });
    const data = await resp.json();
    console.log("Injected:", data.alert_id);
  } catch(e) {
    console.error("Inject failed:", e);
  }
}

document.querySelectorAll("#btn-simulate, #btn-simulate-2").forEach(btn => {
  btn?.addEventListener("click", injectTestAlert);
});

// ─────────────────────────────────────────────────────────────
// CLEAR BUTTON
// ─────────────────────────────────────────────────────────────

$("btn-clear")?.addEventListener("click", async () => {
  if (!confirm("Clear all alerts from dashboard?")) return;
  await fetch("/api/clear", { method: "DELETE" });
  STATE.alerts = [];
  STATE.selectedAlert = null;
  DOM.alertList.innerHTML = "";
  DOM.alertList.appendChild(createEmptyState());
  DOM.reportEmpty.style.display = "";
  DOM.reportContent.style.display = "none";
  DOM.reportBadge.textContent = "AWAITING SELECTION";
  updateStats();
  updateAnalytics();
});

function createEmptyState() {
  const div = document.createElement("div");
  div.className = "empty-state";
  div.id = "empty-state";
  div.innerHTML = `
    <div class="empty-icon">👁</div>
    <p>ARGUS is active. Monitoring feeds...</p>
    <p class="empty-sub">Alerts will appear here in real time.</p>
    <button class="btn-action btn-simulate" id="btn-simulate-2" style="margin-top:16px">⊕ Simulate Alert</button>
  `;
  div.querySelector("#btn-simulate-2")?.addEventListener("click", injectTestAlert);
  return div;
}

// ─────────────────────────────────────────────────────────────
// NARRATIVE TOGGLE
// ─────────────────────────────────────────────────────────────

DOM.btnNarrative?.addEventListener("click", () => {
  const isHidden = DOM.narrativeText.style.display === "none";
  DOM.narrativeText.style.display = isHidden ? "" : "none";
  DOM.btnNarrative.textContent    = isHidden
    ? "▲ HIDE NARRATIVE REPORT"
    : "📄 VIEW FULL NARRATIVE REPORT";
  DOM.btnNarrative.setAttribute("aria-expanded", isHidden ? "true" : "false");
});

// ─────────────────────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────────────────────

function formatTimestamp(ts) {
  if (!ts) return "—";
  try {
    const d = new Date(ts);
    return d.toLocaleString("en-US", {
      month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      hour12: false
    });
  } catch { return ts; }
}

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────

(async function init() {
  // Initial gauge (empty)
  drawThreatGauge(0, "#78909C");

  // Load existing alerts
  await loadExistingAlerts();

  // Load stats
  await loadStats();

  // Connect to SSE stream
  connectSSE();

  // Periodic stats refresh
  STATE.statsInterval = setInterval(loadStats, 10000);

  console.log("[ARGUS] Command Dashboard initialized.");
})();
