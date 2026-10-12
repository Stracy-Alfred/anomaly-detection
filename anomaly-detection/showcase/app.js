/**
 * SMD Anomaly Detection — Interactive Showcase Engine
 */

(function () {
  'use strict';

  // State
  let telemetryData = null;
  let currentSignalKey = 'cpu';
  let currentModel = 'lstm';
  let currentThresholdPercentile = 0.99; // 99.0%
  let isPlaying = false;
  let playAnimId = null;
  let playOffset = 0;
  let hoverIndex = -1;

  // DOM Elements
  const telemetryCanvas = document.getElementById('telemetryCanvas');
  const scoreCanvas = document.getElementById('scoreCanvas');
  const signalSelector = document.getElementById('signal-selector');
  const modelSelector = document.getElementById('model-score-selector');
  const chartSignalTitle = document.getElementById('chart-signal-title');
  const cursorReadout = document.getElementById('cursor-readout');
  const statMaxScore = document.getElementById('stat-max-score');
  const statAlarmCount = document.getElementById('stat-alarm-count');
  const btnPlaySim = document.getElementById('btn-play-sim');
  const playIcon = document.getElementById('play-icon');
  const playText = document.getElementById('play-text');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValDisplay = document.getElementById('threshold-val-display');

  // Confusion matrix DOM
  const cmTP = document.getElementById('cm-tp');
  const cmFP = document.getElementById('cm-fp');
  const cmFN = document.getElementById('cm-fn');
  const cmTN = document.getElementById('cm-tn');

  // Dynamic metrics DOM
  const dynPrecBar = document.getElementById('dyn-prec-bar');
  const dynPrecVal = document.getElementById('dyn-prec-val');
  const dynRecBar = document.getElementById('dyn-rec-bar');
  const dynRecVal = document.getElementById('dyn-rec-val');
  const dynPaf1Bar = document.getElementById('dyn-paf1-bar');
  const dynPaf1Val = document.getElementById('dyn-paf1-val');
  const dynFprBar = document.getElementById('dyn-fpr-bar');
  const dynFprVal = document.getElementById('dyn-fpr-val');

  const attributionContainer = document.getElementById('attribution-container');

  // Bundled fallback data in case of file:// CORS restrictions
  const fallbackData = {
    machine_id: 'machine-1-1',
    total_timesteps: 28479,
    total_anomalies: 2694,
    feature_names: [
      'CPU Utilization %', 'CPU System %', 'CPU User %', 'Memory Used %', 'Memory Free MB',
      'Active Memory MB', 'Inactive Memory MB', 'Buffer Memory MB', 'Cached Memory MB', 'Swap Used MB',
      'Swap Free MB', 'Disk Read IOPS', 'Disk Write IOPS', 'Disk Read KB/s', 'Disk Write KB/s',
      'Disk Queue Length', 'Network Inbound KB/s', 'Network Outbound KB/s', 'Network Inbound Packets', 'Network Outbound Packets',
      'Network Error Packets', 'TCP Active Connections', 'TCP Passive Connections', 'TCP Established', 'TCP Reset Sent',
      'Context Switches/s', 'Interrupts/s', 'Load Avg 1m', 'Load Avg 5m', 'Load Avg 15m',
      'Process Count', 'Running Threads', 'Blocked Threads', 'Page Faults/s', 'Socket Allocations',
      'File Handles Open', 'Entropy Available', 'I/O Wait Time %'
    ],
    signals: {
      cpu: [],
      cpu_recon: [],
      mem: [],
      mem_recon: [],
      disk: [],
      net: [],
      load: [],
      labels: [],
      lstm_scores: [],
      iforest_scores: []
    },
    attribution: [
      { index: 0, name: 'CPU Utilization %', normal_mean_err: 0.05, anomaly_mean_err: 1.84, importance_score: 9.4 },
      { index: 1, name: 'CPU System %', normal_mean_err: 0.04, anomaly_mean_err: 1.62, importance_score: 8.9 },
      { index: 2, name: 'CPU User %', normal_mean_err: 0.06, anomaly_mean_err: 1.45, importance_score: 8.2 },
      { index: 3, name: 'Memory Used %', normal_mean_err: 0.03, anomaly_mean_err: 1.35, importance_score: 7.8 },
      { index: 11, name: 'Disk Read IOPS', normal_mean_err: 0.02, anomaly_mean_err: 1.12, importance_score: 6.9 },
      { index: 12, name: 'Disk Write IOPS', normal_mean_err: 0.04, anomaly_mean_err: 0.95, importance_score: 6.1 },
      { index: 16, name: 'Network Inbound KB/s', normal_mean_err: 0.05, anomaly_mean_err: 1.78, importance_score: 9.1 },
      { index: 17, name: 'Network Outbound KB/s', normal_mean_err: 0.04, anomaly_mean_err: 1.42, importance_score: 7.9 },
      { index: 21, name: 'TCP Active Connections', normal_mean_err: 0.03, anomaly_mean_err: 0.88, importance_score: 5.4 },
      { index: 23, name: 'TCP Established', normal_mean_err: 0.02, anomaly_mean_err: 0.65, importance_score: 4.8 },
      { index: 27, name: 'Load Avg 1m', normal_mean_err: 0.05, anomaly_mean_err: 1.68, importance_score: 8.8 },
      { index: 37, name: 'I/O Wait Time %', normal_mean_err: 0.02, anomaly_mean_err: 1.05, importance_score: 6.4 }
    ]
  };

  // Generate smooth synthetic series for fallback if needed
  function generateFallbackSeries() {
    const N = 750;
    const cpu = [];
    const cpu_recon = [];
    const mem = [];
    const mem_recon = [];
    const disk = [];
    const net = [];
    const load = [];
    const labels = [];
    const lstm_scores = [];
    const iforest_scores = [];

    for (let i = 0; i < N; i++) {
      const isAnomaly1 = i >= 110 && i <= 210;
      const isAnomaly2 = i >= 490 && i <= 560;
      const isAnom = isAnomaly1 || isAnomaly2 ? 1 : 0;
      labels.push(isAnom);

      // Base cyclical signals
      const baseCpu = 0.35 + 0.15 * Math.sin(i * 0.05) + Math.random() * 0.05;
      const baseMem = 0.50 + 0.08 * Math.cos(i * 0.03) + (i / N) * 0.15;
      const baseDisk = 0.20 + 0.10 * Math.sin(i * 0.08 + 1);
      const baseNet = 0.25 + 0.12 * Math.cos(i * 0.04);
      const baseLoad = 0.30 + 0.10 * Math.sin(i * 0.05);

      if (isAnom) {
        const surge = isAnomaly1 ? 0.45 : 0.38;
        cpu.push(Math.min(1.0, baseCpu + surge + Math.random() * 0.1));
        cpu_recon.push(baseCpu + 0.02);
        mem.push(Math.min(1.0, baseMem + (isAnomaly2 ? 0.4 : 0.1)));
        mem_recon.push(baseMem);
        disk.push(Math.min(1.0, baseDisk + 0.4));
        net.push(Math.min(1.0, baseNet + (isAnomaly1 ? 0.55 : 0.2)));
        load.push(Math.min(1.0, baseLoad + 0.45));

        const score = 1.6 + Math.random() * 1.5 + (isAnomaly1 ? 0.6 : 0.2);
        lstm_scores.push(Number(score.toFixed(4)));
        iforest_scores.push(Number((0.68 + Math.random() * 0.25).toFixed(4)));
      } else {
        cpu.push(Number(baseCpu.toFixed(4)));
        cpu_recon.push(Number((baseCpu * 0.98 + Math.random() * 0.02).toFixed(4)));
        mem.push(Number(baseMem.toFixed(4)));
        mem_recon.push(Number((baseMem * 0.99 + Math.random() * 0.01).toFixed(4)));
        disk.push(Number(baseDisk.toFixed(4)));
        net.push(Number(baseNet.toFixed(4)));
        load.push(Number(baseLoad.toFixed(4)));

        const normalScore = 0.1 + Math.random() * 0.35 + 0.1 * Math.sin(i * 0.1);
        lstm_scores.push(Number(normalScore.toFixed(4)));
        iforest_scores.push(Number((0.42 + Math.random() * 0.15).toFixed(4)));
      }
    }

    fallbackData.signals = {
      cpu, cpu_recon, mem, mem_recon, disk, net, load, labels, lstm_scores, iforest_scores
    };
  }

  // Fetch telemetry JSON
  async function loadData() {
    try {
      const resp = await fetch('telemetry_data.json');
      if (!resp.ok) throw new Error('Fetch failed');
      telemetryData = await resp.json();
    } catch (e) {
      console.warn('Using local fallback telemetry data:', e);
      generateFallbackSeries();
      telemetryData = fallbackData;
    }
    initAttributionGrid();
    updateSimulator();
    renderCharts();
  }

  // Render Feature Attribution Grid
  function initAttributionGrid() {
    if (!attributionContainer || !telemetryData) return;
    attributionContainer.innerHTML = '';

    telemetryData.attribution.forEach(attr => {
      const isHigh = attr.anomaly_mean_err > 1.2;
      const isMedium = attr.anomaly_mean_err >= 0.7 && attr.anomaly_mean_err <= 1.2;
      const level = isHigh ? 'high' : isMedium ? 'medium' : 'low';
      const pct = Math.min(100, Math.round((attr.anomaly_mean_err / 2.0) * 100));

      const card = document.createElement('div');
      card.className = 'attr-item';
      card.innerHTML = `
        <div class="attr-header">
          <span class="attr-name">${attr.name}</span>
          <span class="attr-idx">Dim #${attr.index}</span>
        </div>
        <div class="attr-bar-wrap">
          <div class="attr-bar ${level}" style="width: ${pct}%;"></div>
        </div>
        <div class="attr-meta">
          <span>Anomaly Error: <strong>${attr.anomaly_mean_err}</strong></span>
          <span>Baseline: ${attr.normal_mean_err}</span>
        </div>
      `;
      attributionContainer.appendChild(card);
    });
  }

  // Get active threshold value
  function getCurrentThreshold() {
    if (!telemetryData) return 1.588;
    const scores = currentModel === 'lstm' ? telemetryData.signals.lstm_scores : telemetryData.signals.iforest_scores;
    const sorted = [...scores].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(sorted.length * currentThresholdPercentile)));
    return sorted[idx];
  }

  // Render Telemetry & Score Charts on HTML5 Canvas
  function renderCharts() {
    if (!telemetryCanvas || !scoreCanvas || !telemetryData) return;

    const tCtx = telemetryCanvas.getContext('2d');
    const sCtx = scoreCanvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;
    const tWidth = telemetryCanvas.clientWidth;
    const tHeight = telemetryCanvas.clientHeight;
    const sWidth = scoreCanvas.clientWidth;
    const sHeight = scoreCanvas.clientHeight;

    telemetryCanvas.width = tWidth * dpr;
    telemetryCanvas.height = tHeight * dpr;
    scoreCanvas.width = sWidth * dpr;
    scoreCanvas.height = sHeight * dpr;

    tCtx.scale(dpr, dpr);
    sCtx.scale(dpr, dpr);

    const signals = telemetryData.signals;
    const raw = signals[currentSignalKey] || signals.cpu;
    const recon = signals[`${currentSignalKey}_recon`] || signals.cpu_recon || raw;
    const labels = signals.labels;
    const scores = currentModel === 'lstm' ? signals.lstm_scores : signals.iforest_scores;
    const threshold = getCurrentThreshold();

    const N = raw.length;
    const windowSize = 350; // Visible window
    const startIdx = Math.min(N - windowSize, Math.max(0, playOffset));
    const endIdx = Math.min(N, startIdx + windowSize);
    const visibleCount = endIdx - startIdx;

    // ── 1. Telemetry Canvas Drawing ────────────────────────────────────
    tCtx.clearRect(0, 0, tWidth, tHeight);

    // Grid lines
    tCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    tCtx.lineWidth = 1;
    for (let y = 0.2; y <= 0.8; y += 0.2) {
      tCtx.beginPath();
      tCtx.moveTo(0, tHeight * y);
      tCtx.lineTo(tWidth, tHeight * y);
      tCtx.stroke();
    }

    // Shaded Anomaly Ground Truth Regions
    tCtx.fillStyle = 'rgba(249, 115, 22, 0.18)';
    let inAnomaly = false;
    let anomStart = 0;
    for (let i = 0; i < visibleCount; i++) {
      const idx = startIdx + i;
      if (labels[idx] === 1 && !inAnomaly) {
        inAnomaly = true;
        anomStart = (i / visibleCount) * tWidth;
      } else if (labels[idx] === 0 && inAnomaly) {
        inAnomaly = false;
        const anomEnd = (i / visibleCount) * tWidth;
        tCtx.fillRect(anomStart, 0, anomEnd - anomStart, tHeight);
      }
    }
    if (inAnomaly) {
      tCtx.fillRect(anomStart, 0, tWidth - anomStart, tHeight);
    }

    // Draw AE Reconstruction Curve (Purple)
    tCtx.beginPath();
    tCtx.strokeStyle = '#a855f7';
    tCtx.lineWidth = 2;
    for (let i = 0; i < visibleCount; i++) {
      const idx = startIdx + i;
      const x = (i / (visibleCount - 1)) * tWidth;
      const val = recon[idx] !== undefined ? recon[idx] : raw[idx];
      const y = tHeight - (val * (tHeight - 30) + 15);
      if (i === 0) tCtx.moveTo(x, y);
      else tCtx.lineTo(x, y);
    }
    tCtx.stroke();

    // Draw Raw Telemetry Signal (Cyan)
    tCtx.beginPath();
    tCtx.strokeStyle = '#38bdf8';
    tCtx.lineWidth = 2;
    for (let i = 0; i < visibleCount; i++) {
      const idx = startIdx + i;
      const x = (i / (visibleCount - 1)) * tWidth;
      const val = raw[idx];
      const y = tHeight - (val * (tHeight - 30) + 15);
      if (i === 0) tCtx.moveTo(x, y);
      else tCtx.lineTo(x, y);
    }
    tCtx.stroke();

    // Hover Cursor on Telemetry Canvas
    if (hoverIndex >= 0 && hoverIndex < visibleCount) {
      const hx = (hoverIndex / (visibleCount - 1)) * tWidth;
      tCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      tCtx.setLineDash([4, 4]);
      tCtx.beginPath();
      tCtx.moveTo(hx, 0);
      tCtx.lineTo(hx, tHeight);
      tCtx.stroke();
      tCtx.setLineDash([]);
    }

    // ── 2. Score Canvas Drawing ────────────────────────────────────────
    sCtx.clearRect(0, 0, sWidth, sHeight);

    // Score Grid lines
    sCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    sCtx.lineWidth = 1;
    for (let y = 0.25; y <= 0.75; y += 0.25) {
      sCtx.beginPath();
      sCtx.moveTo(0, sHeight * y);
      sCtx.lineTo(sWidth, sHeight * y);
      sCtx.stroke();
    }

    const maxScoreVal = Math.max(...scores) || 3.5;
    const thresholdY = sHeight - ((threshold / maxScoreVal) * (sHeight - 30) + 15);

    // Draw Triggered Alarm Bars
    let alarmCount = 0;
    for (let i = 0; i < visibleCount; i++) {
      const idx = startIdx + i;
      const val = scores[idx];
      if (val > threshold) {
        alarmCount++;
        const x = (i / (visibleCount - 1)) * sWidth;
        sCtx.fillStyle = 'rgba(239, 68, 68, 0.35)';
        sCtx.fillRect(x - 1.5, 0, 3, sHeight);
      }
    }

    // Draw Anomaly Score Area & Line
    sCtx.beginPath();
    for (let i = 0; i < visibleCount; i++) {
      const idx = startIdx + i;
      const x = (i / (visibleCount - 1)) * sWidth;
      const val = scores[idx];
      const y = sHeight - ((val / maxScoreVal) * (sHeight - 30) + 15);
      if (i === 0) sCtx.moveTo(x, y);
      else sCtx.lineTo(x, y);
    }
    sCtx.strokeStyle = '#06b6d4';
    sCtx.lineWidth = 2;
    sCtx.stroke();

    // Draw Threshold Line (Red Dashed)
    sCtx.beginPath();
    sCtx.setLineDash([6, 4]);
    sCtx.strokeStyle = '#ef4444';
    sCtx.lineWidth = 1.5;
    sCtx.moveTo(0, thresholdY);
    sCtx.lineTo(sWidth, thresholdY);
    sCtx.stroke();
    sCtx.setLineDash([]);

    // Threshold Text
    sCtx.fillStyle = '#ef4444';
    sCtx.font = '11px "JetBrains Mono", monospace';
    sCtx.fillText(`Threshold: ${threshold.toFixed(3)}`, 10, Math.max(16, thresholdY - 6));

    // Stats readout update
    if (statMaxScore) statMaxScore.textContent = maxScoreVal.toFixed(2);
    if (statAlarmCount) {
      const alarmPct = ((alarmCount / visibleCount) * 100).toFixed(1);
      statAlarmCount.textContent = `${alarmPct}% (${alarmCount} w)`;
    }
  }

  // Update Sensitivity / Confusion Matrix Simulator
  function updateSimulator() {
    if (!telemetryData) return;
    const scores = currentModel === 'lstm' ? telemetryData.signals.lstm_scores : telemetryData.signals.iforest_scores;
    const labels = telemetryData.signals.labels;
    const threshold = getCurrentThreshold();

    let tp = 0, fp = 0, fn = 0, tn = 0;
    const preds = [];

    for (let i = 0; i < scores.length; i++) {
      const isAlarm = scores[i] > threshold ? 1 : 0;
      preds.push(isAlarm);
      const isActual = labels[i];

      if (isAlarm === 1 && isActual === 1) tp++;
      else if (isAlarm === 1 && isActual === 0) fp++;
      else if (isAlarm === 0 && isActual === 1) fn++;
      else tn++;
    }

    // Point-Adjusted Evaluation
    let paAdjusted = [...preds];
    let inSegment = false;
    let segStart = 0;

    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === 1 && !inSegment) {
        inSegment = true;
        segStart = i;
      } else if (labels[i] === 0 && inSegment) {
        inSegment = false;
        let anyDetected = false;
        for (let j = segStart; j < i; j++) {
          if (preds[j] === 1) { anyDetected = true; break; }
        }
        if (anyDetected) {
          for (let j = segStart; j < i; j++) paAdjusted[j] = 1;
        }
      }
    }

    let paTp = 0, paFp = 0;
    for (let i = 0; i < labels.length; i++) {
      if (paAdjusted[i] === 1 && labels[i] === 1) paTp++;
      else if (paAdjusted[i] === 1 && labels[i] === 0) paFp++;
    }

    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const paPrec = paTp + paFp > 0 ? paTp / (paTp + paFp) : 0;
    const paRec = rec; // Recall is bounded
    const paF1 = (paPrec + paRec) > 0 ? (2 * paPrec * paRec) / (paPrec + paRec) : 0;
    const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;

    // Update UI numbers
    if (thresholdValDisplay) {
      thresholdValDisplay.textContent = `${(currentThresholdPercentile * 100).toFixed(1)}% (${threshold.toFixed(3)})`;
    }

    if (cmTP) cmTP.textContent = tp.toLocaleString();
    if (cmFP) cmFP.textContent = fp.toLocaleString();
    if (cmFN) cmFN.textContent = fn.toLocaleString();
    if (cmTN) cmTN.textContent = tn.toLocaleString();

    if (dynPrecVal) dynPrecVal.textContent = prec.toFixed(4);
    if (dynPrecBar) dynPrecBar.style.width = `${Math.min(100, prec * 100)}%`;

    if (dynRecVal) dynRecVal.textContent = rec.toFixed(4);
    if (dynRecBar) dynRecBar.style.width = `${Math.min(100, rec * 100)}%`;

    if (dynPaf1Val) dynPaf1Val.textContent = paF1.toFixed(4);
    if (dynPaf1Bar) dynPaf1Bar.style.width = `${Math.min(100, paF1 * 100)}%`;

    if (dynFprVal) dynFprVal.textContent = fpr.toFixed(4);
    if (dynFprBar) dynFprBar.style.width = `${Math.min(100, fpr * 100)}%`;
  }

  // Animation Loop for Stream Simulation
  function animateStream() {
    if (!isPlaying || !telemetryData) return;
    playOffset = (playOffset + 2) % (telemetryData.signals.cpu.length - 350);
    renderCharts();
    playAnimId = requestAnimationFrame(animateStream);
  }

  // Event Listeners
  function initEvents() {
    if (signalSelector) {
      signalSelector.addEventListener('change', (e) => {
        currentSignalKey = e.target.value;
        const nameMap = {
          cpu: 'CPU Utilization (%)',
          mem: 'Memory Used (%)',
          disk: 'Disk Read IOPS',
          net: 'Network Inbound (KB/s)',
          load: 'Load Average (1m)'
        };
        if (chartSignalTitle) {
          chartSignalTitle.textContent = `${nameMap[currentSignalKey]}: Raw vs Reconstruction`;
        }
        renderCharts();
      });
    }

    if (modelSelector) {
      modelSelector.addEventListener('change', (e) => {
        currentModel = e.target.value;
        updateSimulator();
        renderCharts();
      });
    }

    if (btnPlaySim) {
      btnPlaySim.addEventListener('click', () => {
        isPlaying = !isPlaying;
        if (isPlaying) {
          if (playIcon) playIcon.textContent = '⏸';
          if (playText) playText.textContent = 'Pause Stream';
          animateStream();
        } else {
          if (playIcon) playIcon.textContent = '▶';
          if (playText) playText.textContent = 'Simulate Stream';
          cancelAnimationFrame(playAnimId);
        }
      });
    }

    if (thresholdSlider) {
      thresholdSlider.addEventListener('input', (e) => {
        currentThresholdPercentile = Number(e.target.value) / 1000;
        updateSimulator();
        renderCharts();
      });
    }

    // Preset buttons for threshold
    document.querySelectorAll('.sim-preset-buttons button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sim-preset-buttons button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const val = Number(btn.getAttribute('data-percentile'));
        if (thresholdSlider) thresholdSlider.value = val;
        currentThresholdPercentile = val / 1000;
        updateSimulator();
        renderCharts();
      });
    });

    // Hover tooltip tracking
    if (telemetryCanvas) {
      telemetryCanvas.addEventListener('mousemove', (e) => {
        const rect = telemetryCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const windowSize = 350;
        hoverIndex = Math.floor((x / rect.width) * windowSize);
        const actualIdx = (playOffset || 0) + hoverIndex;

        if (telemetryData && cursorReadout) {
          const rawVal = telemetryData.signals[currentSignalKey][actualIdx] || 0;
          const scoreVal = telemetryData.signals.lstm_scores[actualIdx] || 0;
          const isAnom = telemetryData.signals.labels[actualIdx] === 1;
          cursorReadout.innerHTML = `Window: <strong>#${actualIdx}</strong> | Val: <strong>${rawVal.toFixed(3)}</strong> | Score: <strong>${scoreVal.toFixed(3)}</strong> | ${isAnom ? '<span style="color:#f87171">ANOMALY</span>' : '<span style="color:#34d399">NORMAL</span>'}`;
        }
        renderCharts();
      });

      telemetryCanvas.addEventListener('mouseleave', () => {
        hoverIndex = -1;
        if (cursorReadout) cursorReadout.textContent = 'Hover timeline to inspect coordinates';
        renderCharts();
      });
    }

    // Code tabs switching
    document.querySelectorAll('.code-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.code-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.code-pane').forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetId = btn.getAttribute('data-target');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');
      });
    });

    // Window resize handler
    window.addEventListener('resize', () => {
      renderCharts();
    });

    // Sync Run button
    const btnReload = document.getElementById('btn-reload-data');
    if (btnReload) {
      btnReload.addEventListener('click', () => {
        loadData();
      });
    }
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', () => {
    initEvents();
    loadData();
  });

})();
