"""
SAGE Dashboard HTML — single-file web interface served by the gateway.

Provides live stats (metabolic state, ATP, GPU, cycles) and a chat interface.
Connects via SSE to /stream for real-time updates.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAGE Dashboard</title>
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
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  header h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
  }

  header .meta {
    display: flex;
    gap: 16px;
    align-items: center;
    font-size: 12px;
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
    grid-template-columns: 240px 1fr 380px;
    height: calc(100vh - 49px);
  }

  /* Left Panel — Avatar + Identity */
  .avatar-panel {
    padding: 20px;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    overflow-y: auto;
  }

  .avatar-wrap {
    position: relative;
    width: 200px;
    height: 200px;
    border-radius: 12px;
    overflow: hidden;
  }

  .avatar-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
  }

  .avatar-wrap::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 16px;
    border: 2px solid var(--state-color);
    box-shadow: 0 0 20px color-mix(in srgb, var(--state-color) 40%, transparent);
    animation: glow 2s ease-in-out infinite;
    pointer-events: none;
  }

  @keyframes glow {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
  }

  .machine-name {
    font-size: 18px;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .lct-id {
    font-size: 10px;
    color: var(--text-dim);
    word-break: break-all;
    text-align: center;
  }

  .metabolic-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: color-mix(in srgb, var(--state-color) 15%, transparent);
    color: var(--state-color);
    border: 1px solid var(--state-color);
  }

  .uptime {
    font-size: 11px;
    color: var(--text-dim);
  }

  .network-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-dim);
    transition: all 0.3s;
    user-select: none;
    font-family: inherit;
    margin-top: 4px;
  }

  .network-toggle:hover {
    border-color: var(--text-dim);
  }

  .network-toggle.open {
    border-color: var(--accent);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, var(--surface));
  }

  .network-toggle .indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
    transition: background 0.3s;
  }

  .network-toggle.open .indicator {
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
  }

  /* Center Panel — Stats */
  .stats-panel {
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }

  .stat-card label {
    display: block;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 6px;
  }

  .stat-card .value {
    font-size: 22px;
    font-weight: 700;
    color: white;
  }

  .stat-card .sub {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 2px;
  }

  .bar-wrap {
    background: #1a1a1a;
    border-radius: 4px;
    height: 12px;
    overflow: hidden;
    margin-top: 6px;
  }

  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease, background 0.5s ease;
  }

  .bar-fill.atp {
    background: linear-gradient(90deg, #ff3333, #ffd700, #00ff41);
    background-size: 300% 100%;
  }

  .bar-fill.gpu {
    background: #4488ff;
  }

  .stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .trust-bars {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 6px;
  }

  .trust-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
  }

  .trust-row .name {
    width: 120px;
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .trust-row .mini-bar {
    flex: 1;
    height: 6px;
    background: #1a1a1a;
    border-radius: 3px;
    overflow: hidden;
  }

  .trust-row .mini-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  .trust-row .val {
    width: 36px;
    text-align: right;
    color: var(--text);
  }

  /* Right Panel — Chat */
  .chat-panel {
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-height: 0;
  }

  .chat-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
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
    max-width: 95%;
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

  .chat-form {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
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

  .chat-form textarea:focus {
    border-color: var(--accent);
  }

  .chat-form textarea:disabled {
    opacity: 0.5;
  }

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
  @media (max-width: 900px) {
    .grid {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr 1fr;
      height: auto;
    }
    .avatar-panel {
      flex-direction: row;
      border-right: none;
      border-bottom: 1px solid var(--border);
      padding: 12px;
    }
    .avatar-wrap { width: 80px; height: 80px; }
    .chat-panel {
      border-left: none;
      border-top: 1px solid var(--border);
      height: 50vh;
    }
  }
</style>
</head>
<body>
  <header>
    <h1>SAGE</h1>
    <div class="meta">
      <span id="cycle-display">Cycle: --</span>
      <span id="uptime-display">Uptime: --</span>
      <span><span class="connection-dot" id="conn-dot"></span> <span id="conn-label">connecting</span></span>
    </div>
  </header>

  <div class="grid">
    <!-- Left: Avatar + Identity -->
    <section class="avatar-panel">
      <div class="avatar-wrap">
        <img src="/images/agentzero.png" alt="SAGE" id="sage-face" />
      </div>
      <div class="machine-name" id="machine-name">--</div>
      <div class="metabolic-badge" id="metabolic-badge">--</div>
      <div class="lct-id" id="lct-id">--</div>
      <div class="uptime" id="uptime-detail">--</div>
      <button class="network-toggle" id="network-toggle" title="Allow others on the network to talk to SAGE">
        <span class="indicator"></span>
        <span id="network-label">Local Only</span>
      </button>
    </section>

    <!-- Center: Stats -->
    <section class="stats-panel">
      <div class="stat-row">
        <div class="stat-card">
          <label>ATP</label>
          <div class="value" id="atp-value">-- / --</div>
          <div class="bar-wrap"><div class="bar-fill atp" id="atp-bar" style="width:0%"></div></div>
        </div>
        <div class="stat-card">
          <label>Cycles</label>
          <div class="value" id="cycle-value">0</div>
          <div class="sub" id="effects-sub">effects: --</div>
        </div>
      </div>

      <div class="stat-row">
        <div class="stat-card">
          <label>GPU Memory</label>
          <div class="value" id="gpu-value">--</div>
          <div class="bar-wrap"><div class="bar-fill gpu" id="gpu-bar" style="width:0%"></div></div>
          <div class="sub" id="gpu-name">--</div>
        </div>
        <div class="stat-card">
          <label>System</label>
          <div class="value" id="cpu-value">--%</div>
          <div class="sub" id="ram-sub">RAM: --</div>
        </div>
      </div>

      <div class="stat-card">
        <label>SNARC Salience</label>
        <div class="value" id="salience-value">0.000</div>
        <div class="sub" id="messages-sub">messages: --</div>
      </div>

      <div class="stat-card">
        <label>Plugin Trust</label>
        <div class="trust-bars" id="trust-bars">
          <div class="sub">waiting for data...</div>
        </div>
      </div>
    </section>

    <!-- Right: Chat -->
    <section class="chat-panel">
      <div class="chat-header">Talk to SAGE</div>
      <div class="chat-log" id="chat-log">
        <div class="chat-msg sage">
          <div class="sender">SAGE</div>
          <div>Dashboard connected. Type a message to begin.</div>
        </div>
      </div>
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
  // Machine + LCT
  if (d.machine) document.getElementById('machine-name').textContent = d.machine;
  if (d.lct_id) document.getElementById('lct-id').textContent = d.lct_id;

  // Metabolic state
  const state = (d.metabolic_state || 'unknown').toLowerCase();
  const badge = document.getElementById('metabolic-badge');
  badge.textContent = state.toUpperCase();
  setStateColor(state);

  // ATP
  if (d.atp_current !== undefined && d.atp_max) {
    const pct = Math.round((d.atp_current / d.atp_max) * 100);
    document.getElementById('atp-value').textContent =
      d.atp_current.toFixed(1) + ' / ' + d.atp_max;
    const bar = document.getElementById('atp-bar');
    bar.style.width = pct + '%';
    bar.style.backgroundPosition = (100 - pct) + '% 0';
  }

  // Cycles
  if (d.cycle_count !== undefined) {
    document.getElementById('cycle-value').textContent = d.cycle_count.toLocaleString();
    document.getElementById('cycle-display').textContent = 'Cycle: ' + d.cycle_count.toLocaleString();
  }

  // Effects
  if (d.loop_stats) {
    const ls = d.loop_stats;
    document.getElementById('effects-sub').textContent =
      'proposed: ' + (ls.effects_proposed || 0) + '  approved: ' + (ls.effects_approved || 0);
  }

  // GPU
  if (d.gpu) {
    const used = d.gpu.memory_allocated_mb;
    const total = d.gpu.memory_total_mb;
    const pct = Math.round((used / total) * 100);
    document.getElementById('gpu-value').textContent = used + ' / ' + total + ' MB';
    document.getElementById('gpu-bar').style.width = pct + '%';
    document.getElementById('gpu-name').textContent = d.gpu.name || '';
  } else {
    document.getElementById('gpu-value').textContent = 'N/A';
    document.getElementById('gpu-name').textContent = d.mode === 'lightweight' ? 'Ollama / CPU' : 'no GPU detected';
  }

  // CPU / RAM
  if (d.cpu_percent !== undefined) {
    document.getElementById('cpu-value').textContent = d.cpu_percent.toFixed(0) + '%';
  }
  if (d.ram_used_mb !== undefined && d.ram_total_mb) {
    document.getElementById('ram-sub').textContent =
      'RAM: ' + Math.round(d.ram_used_mb) + ' / ' + Math.round(d.ram_total_mb) + ' MB';
  }

  // Salience
  if (d.average_salience !== undefined) {
    document.getElementById('salience-value').textContent = d.average_salience.toFixed(3);
  }

  // Messages
  if (d.message_stats) {
    const ms = d.message_stats;
    document.getElementById('messages-sub').textContent =
      'messages: ' + (ms.submitted || 0) + ' in / ' + (ms.resolved || 0) + ' out';
  }

  // Plugin trust
  if (d.plugin_trust && Object.keys(d.plugin_trust).length > 0) {
    const container = document.getElementById('trust-bars');
    container.innerHTML = '';
    const sorted = Object.entries(d.plugin_trust).sort((a, b) => b[1] - a[1]);
    for (const [name, val] of sorted) {
      const shortName = name.replace(/_impl$/, '').replace(/_plugin$/, '').replace(/_irp$/, '');
      const pct = Math.round(val * 100);
      container.innerHTML += '<div class="trust-row">' +
        '<span class="name" title="' + name + '">' + shortName + '</span>' +
        '<div class="mini-bar"><div class="mini-fill" style="width:' + pct + '%"></div></div>' +
        '<span class="val">' + val.toFixed(2) + '</span></div>';
    }
  }

  // Uptime
  if (d.uptime_seconds !== undefined) {
    const h = Math.floor(d.uptime_seconds / 3600);
    const m = Math.floor((d.uptime_seconds % 3600) / 60);
    const s = Math.floor(d.uptime_seconds % 60);
    const str = (h > 0 ? h + 'h ' : '') + m + 'm ' + s + 's';
    document.getElementById('uptime-display').textContent = 'Up: ' + str;
    document.getElementById('uptime-detail').textContent = 'Uptime: ' + str;
  }

  // Chat count for lightweight mode
  if (d.chat_count !== undefined) {
    document.getElementById('messages-sub').textContent = 'chats: ' + d.chat_count;
  }

  // Network access state
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

function appendChat(sender, text, cssClass) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + (cssClass || 'sage');
  div.innerHTML = '<div class="sender">' + escapeHtml(sender) + '</div>' +
                  '<div>' + escapeHtml(text) + '</div>';
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  appendChat('You', message, 'user');
  chatInput.value = '';
  chatInput.disabled = true;
  chatSend.disabled = true;

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sender: 'dashboard@localhost',
        message: message,
        max_wait_seconds: 90,
      }),
    });
    const result = await resp.json();

    if (resp.status === 202) {
      appendChat('SAGE', '(dreaming... message queued, will respond when awake)', 'dream');
    } else if (result.error) {
      appendChat('SAGE', result.error, 'error');
    } else {
      appendChat('SAGE', result.response || result.text || JSON.stringify(result), 'sage');
    }
  } catch (err) {
    appendChat('System', 'Connection error: ' + err.message, 'error');
  }

  chatInput.disabled = false;
  chatSend.disabled = false;
  chatInput.style.height = 'auto';
  chatInput.focus();
});

// Enter to send, Shift+Enter for newline; auto-grow textarea
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
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
connectSSE();
</script>
</body>
</html>"""
