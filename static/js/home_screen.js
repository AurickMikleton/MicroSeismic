/* ── MicroSeismic Classifier — home_screen.js ──────────────────────────── */

const API = "/api/home_screen";

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const uploadZone     = document.getElementById("uploadZone");
const fileInput      = document.getElementById("fileInput");
const uploadIdle     = document.getElementById("uploadIdle");
const previewImg     = document.getElementById("previewImg");
const predictBtn     = document.getElementById("predictBtn");
const resultPanel    = document.getElementById("resultPanel");
const resultBadge    = document.getElementById("resultBadge");
const confValue      = document.getElementById("confValue");
const probBars       = document.getElementById("probBars");

const deviceLabel    = document.getElementById("deviceLabel");
const statusDot      = document.getElementById("statusDot");

const epochSlider    = document.getElementById("epochSlider");
const epochDisplay   = document.getElementById("epochDisplay");
const trainBtn       = document.getElementById("trainBtn");
const progressSection= document.getElementById("progressSection");
const progressLabel  = document.getElementById("progressLabel");
const lossLabel      = document.getElementById("lossLabel");
const progressFill   = document.getElementById("progressFill");
const chartWrap      = document.getElementById("chartWrap");
const lossCanvas     = document.getElementById("lossCanvas");
const doneBanner     = document.getElementById("doneBanner");
const errorBanner    = document.getElementById("errorBanner");

const statEvent      = document.getElementById("statEvent").querySelector(".stat-num");
const statNonEvent   = document.getElementById("statNonEvent").querySelector(".stat-num");
const statTotal      = document.getElementById("statTotal").querySelector(".stat-num");

/* ── State ─────────────────────────────────────────────────────────────── */
let selectedFile     = null;
let pollTimer        = null;
let lossHistory      = [];

/* ── Utilities ─────────────────────────────────────────────────────────── */
function fmt(n)    { return n.toFixed(4); }
function pct(n)    { return (n * 100).toFixed(1) + "%"; }
function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

/* ── Status bar ────────────────────────────────────────────────────────── */
async function loadStatus() {
  try {
    const r    = await fetch(`${API}/status`);
    const data = await r.json();

    const dev = data.device || "cpu";
    deviceLabel.textContent = dev.toUpperCase();
    statusDot.className = "status-dot " + (dev === "cpu" ? "dot-cpu" : "dot-gpu");

    // Dataset stats
    const counts = data.dataset?.counts || {};
    statEvent.textContent    = counts["event"]     ?? "—";
    statNonEvent.textContent = counts["non-event"] ?? "—";
    statTotal.textContent    = data.dataset?.total ?? "—";

    // If training already running (e.g. page reload mid-training)
    const t = data.training;
    if (t?.running) {
      lossHistory = (t.history || []).map(h => h.loss);
      showProgress(t);
      startPolling();
    } else if (t?.trained) {
      lossHistory = (t.history || []).map(h => h.loss);
      drawChart();
      doneBanner.classList.remove("hidden");
    }
  } catch (_) {
    deviceLabel.textContent = "Offline";
    statusDot.className = "status-dot dot-cpu";
  }
}

/* ── Upload / Preview ──────────────────────────────────────────────────── */
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    uploadIdle.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;
  resultPanel.classList.add("hidden");
}

/* ── Predict ───────────────────────────────────────────────────────────── */
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictBtn.disabled  = true;
  predictBtn.textContent = "Classifying…";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const r    = await fetch(`${API}/predict`, { method: "POST", body: form });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    predictBtn.disabled  = false;
    predictBtn.textContent = "Run Classification";
  }
});

function showResult(data) {
  const isEvent = data.prediction === "event";
  resultBadge.textContent  = data.prediction.toUpperCase();
  resultBadge.className    = "result-badge " + (isEvent ? "badge-event" : "badge-nonevent");
  confValue.textContent    = pct(data.confidence);

  // Probability bars
  probBars.innerHTML = "";
  const probs = data.probabilities || {};
  for (const [cls, val] of Object.entries(probs)) {
    const bar = document.createElement("div");
    bar.className = "prob-row";
    bar.innerHTML = `
      <span class="prob-cls">${cls}</span>
      <div class="prob-track">
        <div class="prob-fill ${cls === "event" ? "fill-event" : "fill-nonevent"}"
             style="width: ${clamp(val * 100, 0, 100)}%"></div>
      </div>
      <span class="prob-pct">${pct(val)}</span>
    `;
    probBars.appendChild(bar);
  }
  resultPanel.classList.remove("hidden");
}

/* ── Train ─────────────────────────────────────────────────────────────── */
epochSlider.addEventListener("input", () => {
  epochDisplay.textContent = epochSlider.value;
});

trainBtn.addEventListener("click", async () => {
  const epochs = parseInt(epochSlider.value);
  trainBtn.disabled   = true;
  trainBtn.textContent = "Starting…";
  doneBanner.classList.add("hidden");
  errorBanner.classList.add("hidden");
  lossHistory = [];

  try {
    const r    = await fetch(`${API}/train`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ epochs }),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    startPolling();
  } catch (err) {
    showTrainError(err.message);
    trainBtn.disabled   = false;
    trainBtn.textContent = "Start Training";
  }
});

/* ── Polling ───────────────────────────────────────────────────────────── */
function startPolling() {
  progressSection.style.display = "block";
  chartWrap.style.display       = "block";
  clearInterval(pollTimer);
  pollTimer = setInterval(pollTraining, 1500);
}

async function pollTraining() {
  try {
    const r    = await fetch(`${API}/status`);
    const data = await r.json();
    const t    = data.training;
    if (!t) return;

    showProgress(t);

    if (t.history && t.history.length > lossHistory.length) {
      lossHistory = t.history.map(h => h.loss);
      drawChart();
    }

    if (!t.running) {
      clearInterval(pollTimer);
      trainBtn.disabled   = false;
      trainBtn.textContent = "Start Training";

      if (t.error) {
        showTrainError(t.error);
      } else if (t.trained) {
        doneBanner.classList.remove("hidden");
      }
    }
  } catch (_) {}
}

function showProgress(t) {
  const epoch = t.epoch || 0;
  const total = t.total_epochs || 1;
  const pct   = total ? (epoch / total) * 100 : 0;

  progressLabel.textContent = `Epoch ${epoch} / ${total}`;
  lossLabel.textContent     = t.loss != null ? `loss ${fmt(t.loss)}` : "loss —";
  progressFill.style.width  = clamp(pct, 0, 100) + "%";
  trainBtn.textContent      = t.running ? "Training…" : "Start Training";
}

/* ── Loss Chart ────────────────────────────────────────────────────────── */
function drawChart() {
  if (!lossHistory.length) return;

  const dpr  = window.devicePixelRatio || 1;
  const W    = lossCanvas.offsetWidth  || 400;
  const H    = lossCanvas.offsetHeight || 160;
  lossCanvas.width  = W * dpr;
  lossCanvas.height = H * dpr;

  const ctx  = lossCanvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const PAD  = { top: 16, right: 16, bottom: 28, left: 44 };
  const cW   = W - PAD.left - PAD.right;
  const cH   = H - PAD.top  - PAD.bottom;

  const minL = Math.min(...lossHistory);
  const maxL = Math.max(...lossHistory);
  const rng  = maxL - minL || 1;

  const xOf  = i => PAD.left + (i / (lossHistory.length - 1 || 1)) * cW;
  const yOf  = v => PAD.top  + (1 - (v - minL) / rng) * cH;

  // Grid lines
  ctx.strokeStyle = "rgba(200,200,200,0.4)";
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (i / 4) * cH;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
    // y-axis labels
    const val = maxL - (i / 4) * rng;
    ctx.fillStyle   = "#8a9099";
    ctx.font        = `10px 'Quicksand', sans-serif`;
    ctx.textAlign   = "right";
    ctx.fillText(val.toFixed(3), PAD.left - 6, y + 4);
  }

  // x-axis epoch labels
  ctx.fillStyle = "#8a9099";
  ctx.font      = `10px 'Quicksand', sans-serif`;
  ctx.textAlign = "center";
  const step = Math.max(1, Math.floor(lossHistory.length / 5));
  lossHistory.forEach((_, i) => {
    if (i % step === 0 || i === lossHistory.length - 1) {
      ctx.fillText(i + 1, xOf(i), H - PAD.bottom + 14);
    }
  });

  // Gradient fill under line
  const grad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + cH);
  grad.addColorStop(0,   "rgba(107,175,136,0.35)");
  grad.addColorStop(1,   "rgba(107,175,136,0.01)");
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(lossHistory[0]));
  lossHistory.forEach((v, i) => ctx.lineTo(xOf(i), yOf(v)));
  ctx.lineTo(xOf(lossHistory.length - 1), PAD.top + cH);
  ctx.lineTo(xOf(0), PAD.top + cH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(lossHistory[0]));
  lossHistory.forEach((v, i) => ctx.lineTo(xOf(i), yOf(v)));
  ctx.strokeStyle = "#4A9B6F";
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  ctx.stroke();

  // Dot at latest point
  const li = lossHistory.length - 1;
  ctx.beginPath();
  ctx.arc(xOf(li), yOf(lossHistory[li]), 4, 0, Math.PI * 2);
  ctx.fillStyle = "#1E4D2B";
  ctx.fill();
}

/* ── Error helpers ─────────────────────────────────────────────────────── */
function showTrainError(msg) {
  errorBanner.textContent = "Error: " + msg;
  errorBanner.classList.remove("hidden");
}

function showError(msg) {
  resultBadge.textContent = "Error";
  resultBadge.className   = "result-badge badge-error";
  confValue.textContent   = "";
  probBars.innerHTML      = `<p class="error-text">${msg}</p>`;
  resultPanel.classList.remove("hidden");
}

/* ── Init ──────────────────────────────────────────────────────────────── */
loadStatus();
window.addEventListener("resize", () => { if (lossHistory.length) drawChart(); });