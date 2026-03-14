"""
SAGE Dashboard HTML — single-file web interface served by the gateway.

Provides live stats (metabolic state, ATP, GPU, cycles) and a chat interface.
Connects via SSE to /stream for real-time updates.
Chat history is persisted server-side and loaded on page open.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAGE Dashboard</title>
<link rel="icon" href="data:image/png;base64,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">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0a0a;
    --surface: #111;
    --border: #222;
    --text: #c8c8c8;
    --text-dim: #666;
    --accent: #00ff41;
    --state-wake: #00ff41;
    --state-focus: #ffd700;
    --state-rest: #4488ff;
    --state-dream: #cc44ff;
    --state-crisis: #ff3333;
    --state-lightweight: #4488ff;
    --state-color: var(--state-wake);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  header h1 {
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
  }

  header .meta {
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 11px;
    color: var(--text-dim);
  }

  .connection-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff3333;
    display: inline-block;
    transition: background 0.3s;
  }
  .connection-dot.connected { background: var(--accent); }

  .grid {
    display: grid;
    grid-template-columns: 220px 1fr;
    height: calc(100vh - 37px);
  }

  /* Left Panel — Avatar + Identity + Stats */
  .sidebar {
    padding: 12px;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    overflow-y: auto;
    background: var(--surface);
  }

  .avatar-wrap {
    position: relative;
    width: 120px;
    height: 120px;
    border-radius: 10px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .avatar-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 10px;
  }

  .avatar-wrap::after {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 13px;
    border: 2px solid var(--state-color);
    box-shadow: 0 0 16px color-mix(in srgb, var(--state-color) 40%, transparent);
    animation: glow 2s ease-in-out infinite;
    pointer-events: none;
  }

  @keyframes glow {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }

  .machine-name {
    font-size: 16px;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .lct-id {
    font-size: 9px;
    color: var(--text-dim);
    word-break: break-all;
    text-align: center;
  }

  .metabolic-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: color-mix(in srgb, var(--state-color) 15%, transparent);
    color: var(--state-color);
    border: 1px solid var(--state-color);
  }

  .network-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 10px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text-dim);
    transition: all 0.3s;
    user-select: none;
    font-family: inherit;
  }

  .network-toggle { border-color: #aa3333; color: #aa3333; }
  .network-toggle:hover { border-color: #ff4444; color: #ff4444; }
  .network-toggle.open { border-color: var(--accent); color: var(--accent); }

  .network-toggle .indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #aa3333;
    transition: background 0.3s;
  }
  .network-toggle.open .indicator {
    background: var(--accent);
    box-shadow: 0 0 4px var(--accent);
  }

  /* Compact Stats */
  .stats-section {
    width: 100%;
    margin-top: 4px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .stat-compact {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
  }

  .stat-compact label {
    display: block;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 2px;
  }

  .stat-compact .value {
    font-size: 14px;
    font-weight: 700;
    color: white;
  }

  .stat-compact .sub {
    font-size: 10px;
    color: var(--text-dim);
  }

  .stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .bar-wrap {
    background: #1a1a1a;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
    margin-top: 3px;
  }

  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease, background 0.5s ease;
  }

  .bar-fill.atp {
    background: linear-gradient(90deg, #ff3333, #ffd700, #00ff41);
    background-size: 300% 100%;
  }

  .bar-fill.gpu { background: #4488ff; }

  .trust-bars {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: 3px;
  }

  .trust-row {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
  }

  .trust-row .name {
    width: 70px;
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .trust-row .mini-bar {
    flex: 1;
    height: 4px;
    background: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
  }

  .trust-row .mini-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.5s ease;
  }

  .trust-row .val {
    width: 28px;
    text-align: right;
    color: var(--text);
    font-size: 9px;
  }

  /* Chat Panel — takes most of the window */
  .chat-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-height: 0;
  }

  .chat-header {
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
    background: var(--surface);
    flex-shrink: 0;
  }

  .chat-log {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .chat-msg {
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.5;
    max-width: 85%;
    word-wrap: break-word;
  }

  .chat-msg.user {
    background: #1a2a1a;
    border: 1px solid #2a4a2a;
    align-self: flex-end;
    color: #a0d0a0;
  }

  .chat-msg.sage {
    background: var(--surface);
    border: 1px solid var(--border);
    align-self: flex-start;
  }

  .chat-msg.error {
    background: #2a1a1a;
    border: 1px solid #4a2a2a;
    color: #ff6666;
  }

  .chat-msg.dream {
    background: #2a1a3a;
    border: 1px solid #4a2a5a;
    color: #cc88ff;
    font-style: italic;
  }

  .chat-msg .sender {
    font-weight: 700;
    font-size: 11px;
    margin-bottom: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .chat-msg.user .sender { color: var(--accent); }
  .chat-msg.sage .sender { color: var(--state-color); }

  .chat-msg .time {
    font-size: 9px;
    color: var(--text-dim);
    float: right;
    margin-left: 8px;
  }

  .tool-calls {
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-dim);
    border-top: 1px dashed var(--border);
    padding-top: 4px;
  }
  .tool-calls summary {
    cursor: pointer;
    color: var(--accent);
    font-size: 10px;
  }
  .tool-calls ul {
    margin: 4px 0 0 12px;
    padding: 0;
    list-style: none;
  }
  .tool-calls li {
    margin-bottom: 4px;
    padding: 3px 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 3px;
  }
  .tool-calls code {
    font-size: 10px;
    color: var(--text-dim);
  }

  .chat-form {
    display: flex;
    gap: 8px;
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }

  .chat-form textarea {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    outline: none;
    resize: none;
    min-height: 36px;
    max-height: 120px;
    line-height: 1.4;
    overflow-y: auto;
  }

  .chat-form textarea:focus { border-color: var(--accent); }
  .chat-form textarea:disabled { opacity: 0.5; }

  .chat-form button {
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .chat-form button:hover { opacity: 0.8; }
  .chat-form button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Responsive */
  @media (max-width: 700px) {
    .grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
      height: auto;
    }
    .sidebar {
      flex-direction: row;
      flex-wrap: wrap;
      border-right: none;
      border-bottom: 1px solid var(--border);
      padding: 8px;
      justify-content: center;
    }
    .avatar-wrap { width: 60px; height: 60px; }
    .stats-section { flex-direction: row; flex-wrap: wrap; }
    .stat-compact { flex: 1; min-width: 100px; }
    .chat-panel { height: 70vh; }
  }
</style>
</head>
<body>
  <header>
    <h1>SAGE</h1>
    <div class="meta">
      <span id="version-display">v--</span>
      <span id="cycle-display">Cycle: --</span>
      <span id="uptime-display">Up: --</span>
      <span><span class="connection-dot" id="conn-dot"></span> <span id="conn-label">connecting</span></span>
    </div>
  </header>

  <div class="grid">
    <!-- Left: Avatar + Identity + Compact Stats -->
    <section class="sidebar">
      <div class="avatar-wrap">
        <img src="/images/agentzero.png" alt="SAGE" id="sage-face" />
      </div>
      <div class="machine-name" id="machine-name">--</div>
      <div class="metabolic-badge" id="metabolic-badge">--</div>
      <div class="lct-id" id="lct-id">--</div>
      <button class="network-toggle" id="network-toggle" title="Allow others on the network to talk to SAGE">
        <span class="indicator"></span>
        <span id="network-label">Local Only</span>
      </button>

      <div class="stats-section">
        <div class="stat-row">
          <div class="stat-compact">
            <label>ATP</label>
            <div class="value" id="atp-value">--</div>
            <div class="bar-wrap"><div class="bar-fill atp" id="atp-bar" style="width:0%"></div></div>
          </div>
          <div class="stat-compact">
            <label>Cycles</label>
            <div class="value" id="cycle-value">0</div>
            <div class="sub" id="effects-sub">--</div>
          </div>
        </div>

        <div class="stat-row">
          <div class="stat-compact">
            <label>GPU</label>
            <div class="value" id="gpu-value">--</div>
            <div class="bar-wrap"><div class="bar-fill gpu" id="gpu-bar" style="width:0%"></div></div>
            <div class="sub" id="gpu-name">--</div>
          </div>
          <div class="stat-compact">
            <label>System</label>
            <div class="value" id="cpu-value">--%</div>
            <div class="sub" id="ram-sub">RAM: --</div>
          </div>
        </div>

        <div class="stat-compact">
          <label>SNARC</label>
          <div class="value" id="salience-value">0.000</div>
          <div class="sub" id="messages-sub">messages: --</div>
        </div>

        <div class="stat-compact">
          <label>Tools</label>
          <div class="value" id="tool-count">0</div>
          <div class="sub" id="tool-tier">tier: --</div>
          <div class="sub" id="tool-detail">ok: 0  denied: 0</div>
        </div>

        <div class="stat-compact">
          <label>LLM Pool</label>
          <div class="value" id="llm-pool-count">0</div>
          <div class="sub" id="llm-pool-active">active: --</div>
          <div class="trust-bars" id="llm-pool-bars">
            <div class="sub">waiting...</div>
          </div>
        </div>

        <div class="stat-compact">
          <label>Plugin Trust</label>
          <div class="trust-bars" id="trust-bars">
            <div class="sub">waiting...</div>
          </div>
        </div>

        <div class="stat-compact">
          <label>Sensor Trust</label>
          <div class="trust-bars" id="sensor-trust-bars">
            <div class="sub">waiting...</div>
          </div>
        </div>

        <div class="stat-compact">
          <label>Trust Posture</label>
          <div class="sub" id="posture-label">--</div>
          <div class="sub" id="posture-detail"></div>
        </div>
      </div>
    </section>

    <!-- Right: Chat (main area) -->
    <section class="chat-panel">
      <div class="chat-header">Talk to SAGE</div>
      <div class="chat-log" id="chat-log"></div>
      <form class="chat-form" id="chat-form">
        <textarea id="chat-input" placeholder="Say something..." autocomplete="off" rows="1"></textarea>
        <button type="submit" id="chat-send">Send</button>
      </form>
    </section>
  </div>

<script>
// --- State color mapping ---
const STATE_COLORS = {
  wake: '#00ff41', focus: '#ffd700', rest: '#4488ff',
  dream: '#cc44ff', crisis: '#ff3333', lightweight: '#4488ff',
};

function setStateColor(state) {
  const color = STATE_COLORS[state] || STATE_COLORS.wake;
  document.documentElement.style.setProperty('--state-color', color);
}

// --- SSE Connection ---
let evtSource = null;
function connectSSE() {
  evtSource = new EventSource('/stream');

  evtSource.onopen = () => {
    document.getElementById('conn-dot').classList.add('connected');
    document.getElementById('conn-label').textContent = 'live';
  };

  evtSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) { console.error('SSE parse error:', e); }
  };

  evtSource.onerror = () => {
    document.getElementById('conn-dot').classList.remove('connected');
    document.getElementById('conn-label').textContent = 'reconnecting';
    evtSource.close();
    setTimeout(connectSSE, 3000);
  };
}

// --- Dashboard update ---
function updateDashboard(d) {
  if (d.machine) document.getElementById('machine-name').textContent = d.machine;
  if (d.lct_id) document.getElementById('lct-id').textContent = d.lct_id;
  if (d.code_version) document.getElementById('version-display').textContent = 'v' + d.code_version;

  const state = (d.metabolic_state || 'unknown').toLowerCase();
  document.getElementById('metabolic-badge').textContent = state.toUpperCase();
  setStateColor(state);

  if (d.atp_current !== undefined && d.atp_max) {
    const pct = Math.round((d.atp_current / d.atp_max) * 100);
    document.getElementById('atp-value').textContent =
      d.atp_current.toFixed(0) + ' / ' + d.atp_max;
    const bar = document.getElementById('atp-bar');
    bar.style.width = pct + '%';
    bar.style.backgroundPosition = (100 - pct) + '% 0';
  }

  if (d.cycle_count !== undefined) {
    document.getElementById('cycle-value').textContent = d.cycle_count.toLocaleString();
    document.getElementById('cycle-display').textContent = 'Cycle: ' + d.cycle_count.toLocaleString();
  }

  if (d.loop_stats) {
    const ls = d.loop_stats;
    document.getElementById('effects-sub').textContent =
      'fx: ' + (ls.effects_proposed || 0) + '/' + (ls.effects_approved || 0);
  }

  if (d.gpu) {
    const used = d.gpu.memory_allocated_mb;
    const total = d.gpu.memory_total_mb;
    const pct = Math.round((used / total) * 100);
    document.getElementById('gpu-value').textContent =
      (used / 1000).toFixed(1) + '/' + (total / 1000).toFixed(1) + 'G';
    document.getElementById('gpu-bar').style.width = pct + '%';
    document.getElementById('gpu-name').textContent = d.gpu.name || '';
  } else {
    document.getElementById('gpu-value').textContent = 'N/A';
    document.getElementById('gpu-name').textContent =
      d.mode === 'lightweight' ? 'Ollama' : 'no GPU';
  }

  if (d.cpu_percent !== undefined) {
    document.getElementById('cpu-value').textContent = d.cpu_percent.toFixed(0) + '%';
  }
  if (d.ram_used_mb !== undefined && d.ram_total_mb) {
    document.getElementById('ram-sub').textContent =
      'RAM: ' + (d.ram_used_mb / 1000).toFixed(1) + '/' + (d.ram_total_mb / 1000).toFixed(1) + 'G';
  }

  if (d.average_salience !== undefined) {
    document.getElementById('salience-value').textContent = d.average_salience.toFixed(3);
  }

  if (d.message_stats) {
    const ms = d.message_stats;
    document.getElementById('messages-sub').textContent =
      'in: ' + (ms.submitted || 0) + '  out: ' + (ms.resolved || 0);
  }

  if (d.plugin_trust && Object.keys(d.plugin_trust).length > 0) {
    const container = document.getElementById('trust-bars');
    container.innerHTML = '';
    const sorted = Object.entries(d.plugin_trust).sort((a, b) => a[0].localeCompare(b[0]));
    for (const [name, val] of sorted) {
      const shortName = name.replace(/_impl$/, '').replace(/_plugin$/, '').replace(/_irp$/, '');
      const pct = Math.round(val * 100);
      container.innerHTML += '<div class="trust-row">' +
        '<span class="name" title="' + name + '">' + shortName + '</span>' +
        '<div class="mini-bar"><div class="mini-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="val">' + val.toFixed(2) + '</span></div>';
    }
  }

  if (d.sensor_trust && Object.keys(d.sensor_trust).length > 0) {
    const container = document.getElementById('sensor-trust-bars');
    container.innerHTML = '';
    const sorted = Object.entries(d.sensor_trust).sort((a, b) => a[0].localeCompare(b[0]));
    for (const [name, val] of sorted) {
      const pct = Math.round(val * 100);
      const color = val >= 0.15 ? '#4ec9b0' : '#666';
      container.innerHTML += '<div class="trust-row">' +
        '<span class="name">' + name + '</span>' +
        '<div class="mini-bar"><div class="mini-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
        '<span class="val">' + val.toFixed(2) + '</span></div>';
    }
  }

  if (d.trust_posture) {
    const p = d.trust_posture;
    document.getElementById('posture-label').textContent = p.label +
      ' (conf=' + p.confidence.toFixed(2) + ' asym=' + p.asymmetry.toFixed(2) + ' brd=' + p.breadth.toFixed(2) + ')';
    const restricted = p.effect_restrictions.length > 0 ? 'blocked: ' + p.effect_restrictions.join(', ') : 'no restrictions';
    document.getElementById('posture-detail').textContent =
      'dom: ' + p.dominant_modality + ' | ' + restricted;
  }

  if (d.uptime_seconds !== undefined) {
    const h = Math.floor(d.uptime_seconds / 3600);
    const m = Math.floor((d.uptime_seconds % 3600) / 60);
    const str = (h > 0 ? h + 'h ' : '') + m + 'm';
    document.getElementById('uptime-display').textContent = 'Up: ' + str;
  }

  if (d.chat_count !== undefined) {
    document.getElementById('messages-sub').textContent = 'chats: ' + d.chat_count;
  }

  if (d.tool_stats) {
    const ts = d.tool_stats;
    document.getElementById('tool-count').textContent = ts.total || 0;
    document.getElementById('tool-tier').textContent =
      'tier: ' + (ts.tier || '--') + ' (' + (ts.registered || 0) + ' tools)';
    document.getElementById('tool-detail').textContent =
      'ok: ' + (ts.success || 0) + '  denied: ' + (ts.denied || 0);
  }

  if (d.llm_pool) {
    const lp = d.llm_pool;
    document.getElementById('llm-pool-count').textContent = lp.count || 0;
    document.getElementById('llm-pool-active').textContent =
      'active: ' + (lp.active || '--');
    const barsEl = document.getElementById('llm-pool-bars');
    if (lp.entries && lp.entries.length > 0) {
      let html = '';
      lp.entries.forEach(e => {
        const pct = Math.round(e.trust * 100);
        const color = e.healthy ? (e.model_name === lp.active ? '#4ec9b0' : '#569cd6') : '#888';
        const label = e.model_name.split(':').pop() || e.model_name;
        html += '<div style="display:flex;align-items:center;gap:4px;margin:1px 0">' +
          '<span style="width:48px;font-size:10px;text-align:right;opacity:0.7">' + label + '</span>' +
          '<div style="flex:1;background:#333;border-radius:2px;height:8px">' +
          '<div style="width:' + pct + '%;background:' + color +
          ';border-radius:2px;height:100%"></div></div>' +
          '<span style="width:28px;font-size:10px">' + pct + '%</span></div>';
        });
      barsEl.innerHTML = html;
    }
  }

  if (d.network_open !== undefined) {
    networkOpen = d.network_open;
    updateNetworkToggle();
  }
}

// --- Chat ---
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatLog = document.getElementById('chat-log');

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendChat(sender, text, cssClass, timestamp, toolCalls) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + (cssClass || 'sage');
  const timeStr = timestamp ? '<span class="time">' + formatTime(timestamp) + '</span>' : '';
  let html = '<div class="sender">' + timeStr + escapeHtml(sender) + '</div>' +
             '<div>' + escapeHtml(text) + '</div>';
  // Tool call details (collapsible)
  if (toolCalls && toolCalls.length > 0) {
    html += '<details class="tool-calls"><summary>Tools used (' + toolCalls.length + ')</summary><ul>';
    for (const tc of toolCalls) {
      const status = tc.success ? '&check;' : '&cross;';
      const result = tc.success ? (tc.result || '').substring(0, 200) : (tc.error || 'failed');
      html += '<li><b>' + status + ' ' + escapeHtml(tc.name || '') + '</b>';
      if (tc.arguments) html += ' <code>' + escapeHtml(JSON.stringify(tc.arguments)) + '</code>';
      html += '<br><small>' + escapeHtml(result) + '</small></li>';
    }
    html += '</ul></details>';
  }
  div.innerHTML = html;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Load chat history on startup
async function loadChatHistory() {
  try {
    const resp = await fetch('/chat-history');
    if (!resp.ok) return;
    const messages = await resp.json();
    for (const msg of messages) {
      appendChat(msg.sender, msg.text, msg.css_class, msg.timestamp);
    }
    if (messages.length === 0) {
      appendChat('SAGE', 'Dashboard connected. Type a message to begin.', 'sage');
    }
  } catch (e) {
    appendChat('SAGE', 'Dashboard connected. Type a message to begin.', 'sage');
  }
}

let currentConversationId = null;

async function sendChat() {
  const message = chatInput.value.trim();
  if (!message) return;

  appendChat('You', message, 'user');
  chatInput.value = '';
  chatInput.disabled = true;
  chatSend.disabled = true;
  chatSend.textContent = '...';

  try {
    const payload = {
      sender: 'operator',
      message: message,
      max_wait_seconds: 90,
    };
    if (currentConversationId) {
      payload.conversation_id = currentConversationId;
    }
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const text = await resp.text();
    let result;
    try { result = JSON.parse(text); } catch (pe) {
      appendChat('System', 'Bad response: ' + text.substring(0, 200), 'error');
      return;
    }

    if (result.conversation_id) {
      currentConversationId = result.conversation_id;
    }

    if (resp.status === 202) {
      appendChat('SAGE', '(dreaming... message queued)', 'dream');
    } else if (result.error) {
      appendChat('SAGE', 'Error: ' + result.error, 'error');
    } else {
      appendChat('SAGE', result.response || result.text || JSON.stringify(result), 'sage');
    }
  } catch (err) {
    appendChat('System', 'Connection error: ' + err.message, 'error');
  } finally {
    chatInput.disabled = false;
    chatSend.disabled = false;
    chatSend.textContent = 'Send';
    chatInput.style.height = 'auto';
    chatInput.focus();
  }
}

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  sendChat();
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});
chatSend.addEventListener('click', (e) => {
  e.preventDefault();
  sendChat();
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// --- Network Access Toggle ---
const networkToggle = document.getElementById('network-toggle');
let networkOpen = false;

networkToggle.addEventListener('click', async () => {
  try {
    const resp = await fetch('/network-access', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ open: !networkOpen }),
    });
    const result = await resp.json();
    networkOpen = result.network_open;
    updateNetworkToggle();
  } catch (err) {
    console.error('Network toggle failed:', err);
  }
});

function updateNetworkToggle() {
  const toggle = document.getElementById('network-toggle');
  const label = document.getElementById('network-label');
  if (networkOpen) {
    toggle.classList.add('open');
    label.textContent = 'Network Open';
  } else {
    toggle.classList.remove('open');
    label.textContent = 'Local Only';
  }
}

// --- Init ---
loadChatHistory();
connectSSE();
</script>
</body>
</html>"""
