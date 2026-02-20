"""
McNugget Lightweight Gateway — always-on SAGE presence without PyTorch.

Runs the HTTP gateway with OllamaIRP directly, bypassing the full
consciousness loop (which requires torch). Other SAGE instances and
Claude sessions can discover and talk to McNugget on port 8750.

When PyTorch is installed later, switch to the full sage_daemon.py
for metabolic states, ATP budget, and SNARC salience.

Usage:
    python3 -m sage.gateway.mcnugget_gateway
    SAGE_PORT=9000 python3 -m sage.gateway.mcnugget_gateway
"""

import asyncio
import json
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, Any, Optional

# Direct imports — no torch dependency chain
from sage.gateway.machine_config import get_config, detect_machine

# Load OllamaIRP directly (bypass sage.irp.__init__ which pulls torch)
import importlib.util
_ollama_path = Path(__file__).parent.parent / 'irp' / 'plugins' / 'ollama_irp.py'
_spec = importlib.util.spec_from_file_location('ollama_irp', str(_ollama_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OllamaIRP = _mod.OllamaIRP


class McNuggetHandler(BaseHTTPRequestHandler):
    """HTTP handler for lightweight McNugget gateway."""

    llm: OllamaIRP = None
    config = None
    started_at: float = 0
    chat_count: int = 0

    def do_POST(self):
        if self.path == '/chat' or self.path == '/converse':
            self._handle_chat()
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/status':
            self._handle_status()
        elif self.path == '/peers':
            self._handle_peers()
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Optional[Dict[str, Any]]:
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "Empty request body")
                return None
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        except (ValueError, json.JSONDecodeError) as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return None

    def _handle_chat(self):
        data = self._read_body()
        if data is None:
            return

        message = data.get('message', '').strip()
        if not message:
            self.send_error(400, "Missing 'message' field")
            return

        sender = data.get('sender', f'anonymous@{self.client_address[0]}')
        conversation_id = data.get('conversation_id')

        if self.llm is None:
            self._send_json({
                'error': 'no_llm',
                'message': 'OllamaIRP not loaded',
            }, status=503)
            return

        start = time.time()
        response = self.llm.get_response(message)
        elapsed = time.time() - start

        McNuggetHandler.chat_count += 1

        self._send_json({
            'response': response,
            'sender': sender,
            'conversation_id': conversation_id,
            'machine': self.config.machine_name if self.config else 'mcnugget',
            'model': self.llm.model_name,
            'mode': 'lightweight',  # Signals no consciousness loop
            'generation_time_s': round(elapsed, 2),
        })

    def _handle_health(self):
        self._send_json({
            'status': 'alive',
            'machine': self.config.machine_name if self.config else 'mcnugget',
            'lct_id': self.config.lct_id if self.config else 'mcnugget_sage_lct',
            'mode': 'lightweight',
            'has_llm': self.llm is not None,
            'model': self.llm.model_name if self.llm else None,
            'timestamp': time.time(),
        })

    def _handle_status(self):
        status = {
            'machine': self.config.machine_name if self.config else 'mcnugget',
            'model_size': self.config.model_size if self.config else 'ollama',
            'lct_id': self.config.lct_id if self.config else 'mcnugget_sage_lct',
            'mode': 'lightweight',
            'uptime_seconds': time.time() - self.started_at,
            'has_llm': self.llm is not None,
            'model': self.llm.model_name if self.llm else None,
            'available_models': self.llm.list_models() if self.llm else [],
            'chat_count': self.chat_count,
        }
        self._send_json(status)

    def _handle_peers(self):
        # Known SAGE peers in the collective
        self._send_json({
            'peers': [
                {'name': 'thor', 'host': 'thor.local', 'port': 8750, 'model': '14b'},
                {'name': 'sprout', 'host': 'sprout.local', 'port': 8750, 'model': '0.5b'},
                {'name': 'legion', 'host': 'legion.local', 'port': 8750, 'model': '14b'},
            ],
            'self': {
                'name': 'mcnugget',
                'port': self.config.gateway_port if self.config else 8750,
                'model': self.llm.model_name if self.llm else None,
                'mode': 'lightweight',
            },
        })

    def log_message(self, format, *args):
        if '/health' in (args[0] if args else ''):
            return
        print(f"[McNugget] {args[0] if args else format}")


def main():
    machine = detect_machine()
    if machine != 'mcnugget':
        print(f"Detected machine: {machine} (not mcnugget)")
        print("This gateway is designed for McNugget. Use sage_daemon.py for other machines.")
        print("Or set SAGE_MACHINE=mcnugget to override.")
        sys.exit(1)

    config = get_config(machine)
    port = config.gateway_port

    # Load OllamaIRP
    model_name = config.model_path.split(':', 1)[1] if ':' in config.model_path else 'gemma3:12b'
    print(f"Loading OllamaIRP for {model_name}...")
    try:
        llm = OllamaIRP({
            'model_name': model_name,
            'ollama_host': 'http://localhost:11434',
            'max_response_tokens': config.max_response_tokens,
        })
    except Exception as e:
        print(f"[ERROR] Failed to load OllamaIRP: {e}")
        llm = None

    # Configure handler
    McNuggetHandler.llm = llm
    McNuggetHandler.config = config
    McNuggetHandler.started_at = time.time()

    # Start HTTP server (SO_REUSEADDR to avoid port conflicts on restart)
    import socket
    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True
        allow_reuse_port = True
    httpd = ReusableHTTPServer(('0.0.0.0', port), McNuggetHandler)

    print(f"\n{'='*60}")
    print(f"  McNugget gateway running (lightweight mode)")
    print(f"  Gateway: http://0.0.0.0:{port}")
    print(f"  Model: {model_name} via Ollama")
    print(f"  LCT: {config.lct_id}")
    print(f"  Health: http://localhost:{port}/health")
    print(f"  Note: No consciousness loop (install PyTorch for full daemon)")
    print(f"{'='*60}\n")

    def signal_handler(sig, frame):
        print(f"\n[McNugget] Shutting down...")
        httpd.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        print("[McNugget] Shutdown complete.")


if __name__ == '__main__':
    main()
