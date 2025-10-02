#!/usr/bin/env python3
"""
SAGE Performance Monitoring Dashboard
Real-time metrics visualization for edge deployment
"""

import time
import json
import subprocess
import psutil
from flask import Flask, render_template_string, jsonify
from datetime import datetime
import threading
from collections import deque
from pathlib import Path

class SAGEMonitor:
    """Real-time performance monitoring for SAGE on edge"""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = {
            'timestamp': deque(maxlen=history_size),
            'fps': deque(maxlen=history_size),
            'memory_gb': deque(maxlen=history_size),
            'gpu_util': deque(maxlen=history_size),
            'temp_c': deque(maxlen=history_size),
            'power_w': deque(maxlen=history_size)
        }

        self.current_metrics = {}
        self.alerts = []
        self.monitoring = False
        self.monitor_thread = None

        # Alert thresholds
        self.thresholds = {
            'temp_c': 85.0,
            'memory_gb': 4.0,
            'power_w': 15.0,
            'fps_min': 10.0
        }

    def measure_fps(self) -> float:
        """Measure inference FPS"""
        # This would connect to actual SAGE inference
        # For now, return simulated value
        import random
        return 15.0 + random.uniform(-2, 2)

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0

    def get_temperature(self) -> float:
        """Get CPU/GPU temperature"""
        try:
            # CPU temp
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = int(f.read().strip()) / 1000.0

            # GPU temp
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_temp = float(result.stdout.strip())
                return max(cpu_temp, gpu_temp)

            return cpu_temp
        except:
            return 50.0

    def get_power_draw(self) -> float:
        """Get system power draw in watts"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass

        # Estimate from tegrastats if nvidia-smi fails
        try:
            result = subprocess.run(
                ['timeout', '1', 'tegrastats'],
                capture_output=True, text=True
            )
            # Parse power from tegrastats output
            if 'POM_5V_IN' in result.stdout:
                import re
                match = re.search(r'POM_5V_IN (\d+)', result.stdout)
                if match:
                    return float(match.group(1)) / 1000.0
        except:
            pass

        return 10.0  # Default estimate

    def track_metrics(self) -> Dict:
        """Collect all metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'fps': self.measure_fps(),
            'memory_gb': self.get_memory_usage(),
            'gpu_util': self.get_gpu_utilization(),
            'temp_c': self.get_temperature(),
            'power_w': self.get_power_draw()
        }

        # Check for alerts
        self._check_alerts(metrics)

        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        self.current_metrics = metrics
        return metrics

    def _check_alerts(self, metrics: Dict):
        """Check metrics against thresholds"""
        alerts = []

        if metrics['temp_c'] > self.thresholds['temp_c']:
            alerts.append(f"🔥 Temperature critical: {metrics['temp_c']:.1f}°C")

        if metrics['memory_gb'] > self.thresholds['memory_gb']:
            alerts.append(f"💾 Memory exceeded: {metrics['memory_gb']:.2f}GB")

        if metrics['power_w'] > self.thresholds['power_w']:
            alerts.append(f"⚡ Power exceeded: {metrics['power_w']:.1f}W")

        if metrics['fps'] < self.thresholds['fps_min']:
            alerts.append(f"🐌 FPS below target: {metrics['fps']:.1f}")

        self.alerts = alerts

    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            self.track_metrics()
            time.sleep(1)  # Update every second

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def generate_dashboard(self):
        """Generate web dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>SAGE Edge Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #0f0; padding: 20px; }
        h1 { text-align: center; color: #0f0; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
        .metric { background: #0a0a0a; padding: 15px; border: 1px solid #0f0; border-radius: 5px; }
        .metric h3 { margin: 0 0 10px 0; color: #0f0; }
        .value { font-size: 24px; font-weight: bold; }
        .alert { color: #f00; background: rgba(255,0,0,0.1); padding: 10px; margin: 10px 0; }
        .chart { height: 300px; margin: 20px 0; }
        .good { color: #0f0; }
        .warning { color: #ff0; }
        .critical { color: #f00; }
    </style>
</head>
<body>
    <h1>🌱 SAGE Edge Monitor - Jetson Orin Nano</h1>

    <div class="metrics">
        <div class="metric">
            <h3>📊 FPS</h3>
            <div id="fps" class="value">--</div>
        </div>
        <div class="metric">
            <h3>💾 Memory</h3>
            <div id="memory" class="value">--</div>
        </div>
        <div class="metric">
            <h3>🖥️ GPU</h3>
            <div id="gpu" class="value">--</div>
        </div>
        <div class="metric">
            <h3>🌡️ Temperature</h3>
            <div id="temp" class="value">--</div>
        </div>
        <div class="metric">
            <h3>⚡ Power</h3>
            <div id="power" class="value">--</div>
        </div>
        <div class="metric">
            <h3>⏰ Uptime</h3>
            <div id="uptime" class="value">--</div>
        </div>
    </div>

    <div id="alerts"></div>

    <div id="chart" class="chart"></div>

    <script>
        let startTime = Date.now();

        function updateMetrics() {
            fetch('/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update values
                    document.getElementById('fps').textContent = data.fps.toFixed(1) + ' fps';
                    document.getElementById('fps').className = 'value ' +
                        (data.fps >= 10 ? 'good' : 'critical');

                    document.getElementById('memory').textContent = data.memory_gb.toFixed(2) + ' GB';
                    document.getElementById('memory').className = 'value ' +
                        (data.memory_gb <= 4 ? 'good' : 'critical');

                    document.getElementById('gpu').textContent = data.gpu_util.toFixed(0) + '%';

                    document.getElementById('temp').textContent = data.temp_c.toFixed(1) + '°C';
                    document.getElementById('temp').className = 'value ' +
                        (data.temp_c < 75 ? 'good' : data.temp_c < 85 ? 'warning' : 'critical');

                    document.getElementById('power').textContent = data.power_w.toFixed(1) + 'W';
                    document.getElementById('power').className = 'value ' +
                        (data.power_w <= 15 ? 'good' : 'warning');

                    // Update uptime
                    let uptime = Math.floor((Date.now() - startTime) / 1000);
                    let hours = Math.floor(uptime / 3600);
                    let minutes = Math.floor((uptime % 3600) / 60);
                    let seconds = uptime % 60;
                    document.getElementById('uptime').textContent =
                        hours + 'h ' + minutes + 'm ' + seconds + 's';

                    // Update alerts
                    let alertsDiv = document.getElementById('alerts');
                    if (data.alerts && data.alerts.length > 0) {
                        alertsDiv.innerHTML = data.alerts.map(a =>
                            '<div class="alert">' + a + '</div>').join('');
                    } else {
                        alertsDiv.innerHTML = '';
                    }
                });
        }

        function updateChart() {
            fetch('/history')
                .then(response => response.json())
                .then(data => {
                    let traces = [
                        {y: data.fps, name: 'FPS', yaxis: 'y'},
                        {y: data.temp_c, name: 'Temp °C', yaxis: 'y2'},
                        {y: data.power_w, name: 'Power W', yaxis: 'y3'}
                    ];

                    let layout = {
                        title: 'Performance History',
                        paper_bgcolor: '#1a1a1a',
                        plot_bgcolor: '#0a0a0a',
                        font: {color: '#0f0'},
                        yaxis: {title: 'FPS', side: 'left', color: '#0f0'},
                        yaxis2: {title: 'Temp °C', overlaying: 'y', side: 'right', color: '#ff0'},
                        yaxis3: {title: 'Power W', overlaying: 'y', side: 'right', position: 0.85, color: '#0ff'}
                    };

                    Plotly.newPlot('chart', traces, layout);
                });
        }

        // Update every second
        setInterval(updateMetrics, 1000);
        setInterval(updateChart, 5000);

        // Initial load
        updateMetrics();
        updateChart();
    </script>
</body>
</html>
        """

# Flask web server
app = Flask(__name__)
monitor = SAGEMonitor()

@app.route('/')
def dashboard():
    return monitor.generate_dashboard()

@app.route('/metrics')
def metrics():
    return jsonify(monitor.current_metrics)

@app.route('/history')
def history():
    return jsonify({
        'fps': list(monitor.metrics_history['fps']),
        'temp_c': list(monitor.metrics_history['temp_c']),
        'power_w': list(monitor.metrics_history['power_w'])
    })

def main():
    """Run the monitoring dashboard"""
    print("🚀 SAGE Performance Monitor")
    print("   Starting monitoring...")

    monitor.start_monitoring()

    print("   Dashboard: http://localhost:5000")
    print("\nPress Ctrl+C to stop")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n   Stopping monitor...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
