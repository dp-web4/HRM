"""
SAGE Consciousness Daemon — always-on consciousness loop with HTTP gateway.

Loads the LLM model once, runs SAGEConsciousness.run() continuously,
and exposes an HTTP gateway for external entities (Claude, other SAGEs)
to send messages into the consciousness loop.

Usage:
    # Auto-detect machine and start
    python3 -m sage.gateway.sage_daemon

    # Explicit machine
    SAGE_MACHINE=thor python3 -m sage.gateway.sage_daemon

    # Custom port
    SAGE_PORT=9000 python3 -m sage.gateway.sage_daemon

Architecture:
    ┌─────────────┐     ┌────────────────────┐
    │ HTTP Gateway │────►│ MessageQueue       │
    │ (thread)     │     │ (thread-safe)      │
    └─────────────┘     └────────┬───────────┘
                                 │ poll()
                        ┌────────▼───────────┐
                        │ SAGEConsciousness   │
                        │ .run() (async loop) │
                        └────────┬───────────┘
                                 │ resolve()
                        ┌────────▼───────────┐
                        │ NetworkEffector     │
                        │ → MessageQueue      │
                        └────────────────────┘
"""

import asyncio
import signal
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add sage to path
_sage_root = Path(__file__).parent.parent
_hrm_root = _sage_root.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))
if str(_hrm_root) not in sys.path:
    sys.path.insert(0, str(_hrm_root))

from sage.gateway.machine_config import SAGEMachineConfig, get_config, detect_machine
from sage.gateway.message_queue import MessageQueue


class SAGEDaemon:
    """
    Always-on SAGE consciousness daemon.

    Manages the lifecycle of:
    - LLM model (loaded once, stays in GPU memory)
    - SAGEConsciousness loop (runs continuously)
    - GatewayServer (HTTP server for external communication)
    - MessageQueue (bridge between gateway and consciousness loop)
    """

    def __init__(self, config: Optional[SAGEMachineConfig] = None):
        self.config = config or get_config()
        self.message_queue = MessageQueue()
        self.consciousness = None
        self.gateway = None
        self.llm_plugin = None
        self.started_at = None
        self._shutdown_event = asyncio.Event()

        print(f"SAGE Daemon initializing on {self.config.machine_name}")
        print(f"  Model: {self.config.model_size} at {self.config.model_path}")
        print(f"  Device: {self.config.device}")
        print(f"  Gateway port: {self.config.gateway_port}")

    def _load_llm(self):
        """Load the LLM model into GPU memory. Called once at startup."""
        if not self.config.model_path:
            print(f"[WARN] No model path configured for {self.config.machine_name}. "
                  f"Running without LLM (gateway-only mode).")
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(self.config.workspace_path) / 'HRM' / 'model-zoo' / 'sage' / 'epistemic-stances',
            ]
            found = False
            for alt in alt_paths:
                if alt.exists():
                    # Look for any model directory
                    for candidate in sorted(alt.rglob('config.json')):
                        print(f"[INFO] Found model at: {candidate.parent}")
                        model_path = candidate.parent
                        self.config.model_path = str(model_path)
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"[WARN] Model path not found: {self.config.model_path}")
                print(f"  Running without LLM. Messages will get mock responses.")
                return

        print(f"\nLoading LLM from {self.config.model_path}...")
        start = time.time()

        try:
            if self.config.model_size in ('14b', '30b'):
                from sage.core.multi_model_loader import create_thor_loader, TaskComplexity
                self.llm_plugin = create_thor_loader(preload_default=True)
                print(f"  Multi-model loader ready ({time.time() - start:.1f}s)")
            else:
                # 0.5B — use IntrospectiveQwenIRP directly
                from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP
                self.llm_plugin = IntrospectiveQwenIRP({
                    'model_path': self.config.model_path,
                    'is_merged_model': True,
                    'force_cpu': self.config.device == 'cpu',
                })
                print(f"  IntrospectiveQwenIRP loaded ({time.time() - start:.1f}s)")

        except Exception as e:
            print(f"[ERROR] Failed to load LLM: {e}")
            print(f"  Running without LLM. Messages will get mock responses.")
            self.llm_plugin = None

    def _create_consciousness(self):
        """Create and configure the SAGEConsciousness instance."""
        from sage.core.sage_consciousness import SAGEConsciousness

        consciousness_config = {
            'modalities': ['vision', 'language', 'audio', 'memory'],
            'device': self.config.device,
            'max_atp': 100.0,
            'circadian_period': 100,  # cycles per day in simulation
        }

        # Enable policy gate if available
        try:
            from sage.irp.plugins.policy_gate import PolicyGateIRP
            consciousness_config['enable_policy_gate'] = True
        except ImportError:
            pass

        self.consciousness = SAGEConsciousness(
            config=consciousness_config,
            initial_atp=100.0,
            enable_circadian=True,
            simulation_mode=False,  # Real wall-clock time for daemon
            message_queue=self.message_queue,
            llm_plugin=self.llm_plugin,
        )

        print(f"  Consciousness loop created (simulation_mode=False)")

    def _start_gateway(self):
        """Start the HTTP gateway server."""
        from sage.gateway.gateway_server import GatewayServer

        self.gateway = GatewayServer(
            message_queue=self.message_queue,
            consciousness=self.consciousness,
            config=self.config,
            host='0.0.0.0',
            port=self.config.gateway_port,
        )
        self.gateway.start()
        print(f"  Gateway server started on 0.0.0.0:{self.config.gateway_port}")

    async def start(self):
        """Start the SAGE daemon — load model, create loop, start gateway, run forever."""
        self.started_at = time.time()

        # Set event loop for message queue Futures
        self.message_queue.set_event_loop(asyncio.get_event_loop())

        # 1. Load LLM (stays in memory)
        self._load_llm()

        # 2. Create consciousness loop
        self._create_consciousness()

        # 3. Start HTTP gateway
        self._start_gateway()

        # 4. Print ready banner
        print(f"\n{'='*60}")
        print(f"  SAGE daemon running on {self.config.machine_name}")
        print(f"  Gateway: http://0.0.0.0:{self.config.gateway_port}")
        print(f"  Model: {self.config.model_size}")
        print(f"  LCT: {self.config.lct_id}")
        print(f"  Health: http://localhost:{self.config.gateway_port}/health")
        print(f"{'='*60}\n")

        # 5. Run consciousness loop until shutdown
        try:
            await self.consciousness.run()
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Clean shutdown — persist state, stop gateway, unload model."""
        print(f"\n[SAGE] Shutting down...")

        # Stop consciousness loop
        if self.consciousness:
            self.consciousness.running = False

        # Stop gateway
        if self.gateway:
            self.gateway.stop()
            print(f"  Gateway stopped")

        # Persist state
        self._persist_state()

        # Unload model
        if self.llm_plugin is not None:
            del self.llm_plugin
            self.llm_plugin = None

            # Free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            print(f"  Model unloaded")

        uptime = time.time() - (self.started_at or time.time())
        print(f"  Uptime: {uptime/3600:.1f} hours")
        print(f"  Messages processed: {self.message_queue.stats}")
        print(f"[SAGE] Shutdown complete.")

    def _persist_state(self):
        """Persist consciousness state to disk."""
        if not self.config.identity_state_path:
            return

        try:
            # Save consciousness stats
            if self.consciousness:
                stats_path = Path(self.config.identity_state_path).parent / 'daemon_state.json'
                state = {
                    'last_shutdown': time.time(),
                    'machine': self.config.machine_name,
                    'uptime_seconds': time.time() - (self.started_at or time.time()),
                    'cycles_completed': self.consciousness.cycle_count,
                    'metabolic_state': self.consciousness.metabolic.current_state.value,
                    'atp_level': self.consciousness.metabolic.atp,
                    'message_stats': self.message_queue.stats,
                }
                with open(stats_path, 'w') as f:
                    json.dump(state, f, indent=2)
                print(f"  State persisted to {stats_path}")
        except Exception as e:
            print(f"  [WARN] Failed to persist state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status for health checks."""
        status = {
            'machine': self.config.machine_name,
            'model_size': self.config.model_size,
            'lct_id': self.config.lct_id,
            'uptime_seconds': time.time() - (self.started_at or time.time()),
            'has_llm': self.llm_plugin is not None,
            'message_stats': self.message_queue.stats,
        }
        if self.consciousness:
            status['metabolic_state'] = self.consciousness.metabolic.current_state.value
            status['atp_level'] = self.consciousness.metabolic.atp
            status['cycle_count'] = self.consciousness.cycle_count
        return status


def main():
    """Entry point for the SAGE daemon."""
    machine = detect_machine()
    print(f"Detected machine: {machine}")

    if machine == 'unknown':
        print("Could not detect machine. Set SAGE_MACHINE environment variable.")
        print("  export SAGE_MACHINE=thor    # or sprout, legion, cbp")
        sys.exit(1)

    config = get_config(machine)

    if config.model_size == 'none':
        print(f"{machine} has no local SAGE model. Use cli_client.py to connect to a remote SAGE.")
        sys.exit(0)

    daemon = SAGEDaemon(config)

    # Set up signal handlers for clean shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        print(f"\n[SAGE] Received signal {sig}, initiating shutdown...")
        loop.create_task(daemon.shutdown())
        # Give shutdown 10 seconds then force exit
        loop.call_later(10, sys.exit, 0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(daemon.start())
    except KeyboardInterrupt:
        loop.run_until_complete(daemon.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
