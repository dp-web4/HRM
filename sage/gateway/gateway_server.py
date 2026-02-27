"""
SAGE Gateway HTTP Server — external communication interface.

Provides HTTP endpoints for sending messages to a running SAGE consciousness
loop and receiving responses. Follows the FederationServer pattern from
sage/federation/federation_service.py.

Endpoints:
    POST /chat      — Send message, receive response (blocking)
    POST /converse   — Multi-turn conversation (includes conversation_id)
    GET  /           — SAGE dashboard (live stats + chat)
    GET  /stream     — SSE stream of live stats (1Hz)
    GET  /health     — Health check + metabolic state
    GET  /status     — Full daemon status
    GET  /peers      — List known peer SAGEs
    GET  /images/*   — Static image serving (Agent Zero avatar)

Auth:
    Localhost: No auth required
    Remote (10.0.0.x): Ed25519 signature in X-Signature header
"""

import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, Any, Optional
import concurrent.futures

from sage.gateway.message_queue import MessageQueue


class GatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the SAGE gateway."""

    # Class-level attributes set by GatewayServer
    message_queue: MessageQueue = None
    consciousness = None
    daemon = None
    config = None

    def do_POST(self):
        if self.path == '/chat':
            self._handle_chat()
        elif self.path == '/converse':
            self._handle_chat()  # Same handler, conversation_id distinguishes
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        if self.path in ('/', '/dashboard'):
            self._handle_dashboard()
        elif self.path == '/stream':
            self._handle_stream()
        elif self.path.startswith('/images/'):
            self._handle_static()
        elif self.path == '/health':
            self._handle_health()
        elif self.path == '/status':
            self._handle_status()
        elif self.path == '/peers':
            self._handle_peers()
        else:
            self.send_error(404, "Endpoint not found")

    def _is_localhost(self) -> bool:
        """Check if request is from localhost."""
        client_ip = self.client_address[0]
        return client_ip in ('127.0.0.1', '::1', 'localhost')

    def _check_auth(self) -> bool:
        """
        Check authentication for non-localhost requests.

        Localhost: always allowed (no auth needed).
        Remote: requires Ed25519 signature in X-Signature header.
        """
        if self._is_localhost():
            return True

        # Remote requests need Ed25519 auth
        signature = self.headers.get('X-Signature')
        platform = self.headers.get('X-Platform')

        if not signature or not platform:
            self.send_error(403, "Missing X-Signature or X-Platform header")
            return False

        # TODO: Verify Ed25519 signature against known platform keys
        # For now, accept any request from 10.0.0.x (LAN)
        client_ip = self.client_address[0]
        if client_ip.startswith('10.0.0.'):
            return True

        self.send_error(403, "Unauthorized")
        return False

    def _read_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body."""
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

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_chat(self):
        """Handle POST /chat — send message to SAGE, wait for response."""
        if not self._check_auth():
            return

        data = self._read_body()
        if data is None:
            return

        # Validate required fields
        message = data.get('message', '').strip()
        if not message:
            self.send_error(400, "Missing 'message' field")
            return

        sender = data.get('sender', f'anonymous@{self.client_address[0]}')
        conversation_id = data.get('conversation_id')
        max_wait = min(data.get('max_wait_seconds', 30), 120)  # Cap at 2 min

        # Check if SAGE is dreaming
        if self.consciousness and hasattr(self.consciousness, 'metabolic'):
            from sage.core.metabolic_controller import MetabolicState
            if self.consciousness.metabolic.current_state == MetabolicState.DREAM:
                self._send_json({
                    'status': 'dreaming',
                    'message': 'SAGE is dreaming. Message queued for when it wakes.',
                    'metabolic_state': 'dream',
                    'conversation_id': conversation_id,
                }, status=202)
                # Still submit the message — it will queue
                # But don't wait for a response
                try:
                    self.message_queue.submit(sender, message, conversation_id,
                                              metadata=data.get('metadata'))
                except Exception:
                    pass
                return

        # Submit message to queue
        try:
            future = self.message_queue.submit(
                sender=sender,
                content=message,
                conversation_id=conversation_id,
                metadata=data.get('metadata'),
            )
        except RuntimeError as e:
            self.send_error(500, f"Message queue error: {e}")
            return

        # Wait for response (blocking the HTTP thread)
        try:
            # Run the async future in a synchronous context
            loop = self.message_queue._loop
            result = self._wait_for_future(future, loop, timeout=max_wait)

            if result is None:
                self._send_json({
                    'error': 'timeout',
                    'message': f'No response within {max_wait}s',
                    'conversation_id': conversation_id,
                }, status=504)
                return

            if result.get('error'):
                self._send_json(result, status=504)
                return

            # Add metabolic context to response
            if self.consciousness and hasattr(self.consciousness, 'metabolic'):
                result['metabolic_state'] = self.consciousness.metabolic.current_state.value
                result['atp_remaining'] = round(self.consciousness.metabolic.atp_current, 1)

            self._send_json(result)

        except Exception as e:
            self.send_error(500, f"Error processing message: {e}")

    def _wait_for_future(self, future: asyncio.Future,
                          loop: asyncio.AbstractEventLoop,
                          timeout: float) -> Optional[Dict]:
        """Wait for an async Future from a synchronous thread."""
        # Use a concurrent.futures event to bridge async→sync
        result_container = [None]
        done_event = concurrent.futures.Future()

        def on_done(fut):
            try:
                result_container[0] = fut.result()
            except Exception as e:
                result_container[0] = {'error': str(e)}
            done_event.set_result(True)

        loop.call_soon_threadsafe(future.add_done_callback, on_done)

        try:
            done_event.result(timeout=timeout)
            return result_container[0]
        except concurrent.futures.TimeoutError:
            return None

    def _handle_dashboard(self):
        """Serve the SAGE dashboard web interface."""
        from sage.gateway.dashboard_html import DASHBOARD_HTML
        body = DASHBOARD_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_stream(self):
        """SSE stream of live SAGE stats — pushes JSON every 1 second."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                data = self._collect_dashboard_stats()
                payload = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected

    def _collect_dashboard_stats(self) -> Dict[str, Any]:
        """Gather all stats the dashboard needs."""
        stats: Dict[str, Any] = {
            'timestamp': time.time(),
            'machine': getattr(self.config, 'machine_name', 'unknown') if self.config else 'unknown',
            'lct_id': getattr(self.config, 'lct_id', '') if self.config else '',
        }

        # Metabolic state + ATP
        if self.consciousness and hasattr(self.consciousness, 'metabolic'):
            stats['metabolic_state'] = self.consciousness.metabolic.current_state.value
            stats['atp_current'] = round(self.consciousness.metabolic.atp_current, 1)
            stats['atp_max'] = getattr(self.consciousness.metabolic, 'atp_max',
                                       getattr(self.consciousness.metabolic, 'max_atp', 100.0))
            stats['cycle_count'] = getattr(self.consciousness, 'cycle_count', 0)

        # Consciousness loop stats
        if self.consciousness and hasattr(self.consciousness, 'stats'):
            stats['loop_stats'] = self.consciousness.stats
            stats['average_salience'] = self.consciousness.stats.get('average_salience', 0.0)

        # Plugin trust weights
        if self.consciousness and hasattr(self.consciousness, 'plugin_trust_weights'):
            stats['plugin_trust'] = dict(self.consciousness.plugin_trust_weights)

        # GPU stats (guarded import)
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                stats['gpu'] = {
                    'name': torch.cuda.get_device_name(0),
                    'memory_allocated_mb': round(torch.cuda.memory_allocated(0) / 1e6, 1),
                    'memory_reserved_mb': round(torch.cuda.memory_reserved(0) / 1e6, 1),
                    'memory_total_mb': round(props.total_memory / 1e6, 1),
                }
        except ImportError:
            pass

        # CPU / RAM (guarded import)
        try:
            import psutil
            stats['cpu_percent'] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            stats['ram_used_mb'] = round(vm.used / 1e6, 1)
            stats['ram_total_mb'] = round(vm.total / 1e6, 1)
        except ImportError:
            pass

        # Message queue stats
        if self.message_queue and hasattr(self.message_queue, 'stats'):
            stats['message_stats'] = self.message_queue.stats

        # Uptime
        if self.daemon and hasattr(self.daemon, 'started_at') and self.daemon.started_at:
            stats['uptime_seconds'] = round(time.time() - self.daemon.started_at, 1)

        return stats

    def _handle_static(self):
        """Serve static files from the HRM/images/ directory."""
        filename = self.path.split('/images/')[-1]
        if '..' in filename or '/' in filename:
            self.send_error(403, "Forbidden")
            return

        images_dir = Path(__file__).parent.parent.parent / 'images'
        filepath = images_dir / filename

        if not filepath.exists() or not filepath.is_file():
            self.send_error(404, "File not found")
            return

        ext = filepath.suffix.lower()
        content_types = {
            '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.svg': 'image/svg+xml',
        }
        content_type = content_types.get(ext)
        if not content_type:
            self.send_error(403, "File type not allowed")
            return

        body = filepath.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()
        self.wfile.write(body)

    def _handle_health(self):
        """Handle GET /health — quick health check."""
        health = {
            'status': 'alive',
            'timestamp': time.time(),
        }

        if self.config:
            health['machine'] = self.config.machine_name
            health['lct_id'] = self.config.lct_id

        if self.consciousness and hasattr(self.consciousness, 'metabolic'):
            health['metabolic_state'] = self.consciousness.metabolic.current_state.value
            health['atp_level'] = round(self.consciousness.metabolic.atp_current, 1)
            health['cycle_count'] = self.consciousness.cycle_count

        self._send_json(health)

    def _handle_status(self):
        """Handle GET /status — full daemon status."""
        if self.daemon:
            status = self.daemon.get_status()
        else:
            status = {'error': 'daemon not available'}

        self._send_json(status)

    def _handle_peers(self):
        """Handle GET /peers — list known peer SAGEs."""
        # TODO: Implement peer registry
        peers = {
            'peers': [],
            'note': 'Peer registry not yet implemented',
        }
        self._send_json(peers)

    def log_message(self, format, *args):
        """Custom log formatting — suppress noisy endpoints."""
        path = args[0] if args else ''
        if any(s in path for s in ('/health', '/stream', '/images/')):
            return
        print(f"[Gateway] {args[0] if args else format}")


class GatewayServer:
    """
    HTTP gateway server for SAGE daemon.

    Runs in a background thread. Accepts messages via HTTP and injects
    them into the consciousness loop via MessageQueue.
    """

    def __init__(
        self,
        message_queue: MessageQueue,
        consciousness=None,
        config=None,
        daemon=None,
        host: str = '0.0.0.0',
        port: int = 8750,
    ):
        self.message_queue = message_queue
        self.consciousness = consciousness
        self.config = config
        self.daemon = daemon
        self.host = host
        self.port = port
        self.httpd = None
        self.server_thread = None
        self.running = False

        # Configure handler class variables
        GatewayHandler.message_queue = message_queue
        GatewayHandler.consciousness = consciousness
        GatewayHandler.config = config
        GatewayHandler.daemon = daemon

    def start(self):
        """Start the gateway server in a background thread."""
        if self.running:
            return

        self.httpd = ThreadingHTTPServer((self.host, self.port), GatewayHandler)
        self.running = True

        self.server_thread = Thread(target=self._run, daemon=True, name='sage-gateway')
        self.server_thread.start()

    def _run(self):
        """Server loop — threaded so SSE connections don't block other requests."""
        self.httpd.serve_forever()

    def stop(self):
        """Stop the gateway server."""
        if not self.running:
            return

        self.running = False
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


if __name__ == "__main__":
    import asyncio

    async def test_gateway():
        mq = MessageQueue()
        mq.set_event_loop(asyncio.get_event_loop())

        server = GatewayServer(
            message_queue=mq,
            host='127.0.0.1',
            port=8751,  # Test port
        )
        server.start()
        print(f"Gateway test server running on 127.0.0.1:8751")

        # Test health endpoint
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:8751/health', timeout=2) as r:
                health = json.loads(r.read().decode())
                print(f"Health: {health}")
                assert health['status'] == 'alive'
                print("Health check passed!")
        except Exception as e:
            print(f"Health check failed: {e}")

        # Test chat with a mock responder
        async def mock_responder():
            """Simulate consciousness loop responding to messages."""
            await asyncio.sleep(0.1)
            msg = mq.poll()
            if msg:
                mq.resolve(msg.message_id, f"Echo: {msg.content}",
                           extra={'metabolic_state': 'wake'})

        # Submit via HTTP in a thread
        import threading

        def send_chat():
            data = json.dumps({
                'sender': 'test@local',
                'message': 'Hello SAGE!',
                'max_wait_seconds': 5,
            }).encode()
            req = urllib.request.Request(
                'http://127.0.0.1:8751/chat',
                data=data,
                headers={'Content-Type': 'application/json'},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    response = json.loads(r.read().decode())
                    print(f"Chat response: {response}")
            except Exception as e:
                print(f"Chat failed: {e}")

        chat_thread = threading.Thread(target=send_chat)
        chat_thread.start()

        # Run mock responder
        await mock_responder()
        chat_thread.join(timeout=5)

        server.stop()
        print("\nGateway test complete!")

    asyncio.run(test_gateway())
