"""
DaemonIRP — Thin HTTP adapter that delegates to the resident SAGE daemon.

Instead of loading a model into the script's process, this plugin sends
prompts to the always-on daemon's /chat endpoint and returns the response.
This lets raising scripts swap one import line (IntrospectiveQwenIRP → DaemonIRP)
without rewriting their generation logic.

The daemon loads the model once at startup and keeps it resident. All scripts
share the same loaded model through this HTTP adapter.

Usage:
    from sage.irp.plugins.daemon_irp import DaemonIRP

    model = DaemonIRP({
        'daemon_host': 'localhost',
        'daemon_port': 8750,
        'system_prompt': '...',  # Optional
        'max_wait_seconds': 60,
    })

    # Same interface as IntrospectiveQwenIRP
    state = model.init_state({'prompt': 'Hello', 'memory': []})
    state = model.step(state)
    response = state['current_response']
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional


class DaemonIRP:
    """
    IRP plugin that delegates to the resident SAGE daemon via HTTP.

    Implements the same interface as IntrospectiveQwenIRP:
    - init_state(context) → state dict
    - step(state) → updated state dict
    - halt(state) → bool
    - get_response(state) → str
    - health_check() → dict
    """

    def __init__(self, config: Dict[str, Any]):
        self.host = config.get('daemon_host', 'localhost')
        self.port = config.get('daemon_port', 8750)
        self.base_url = f"http://{self.host}:{self.port}"
        self.system_prompt = config.get('system_prompt', '')
        self.max_wait = config.get('max_wait_seconds', 120)
        self.sender = config.get('sender', 'raising_session')
        self.conversation_id = config.get('conversation_id', None)
        self.temperature = config.get('temperature', 0.7)
        self.max_new_tokens = config.get('max_new_tokens', 150)

        # Verify daemon is reachable
        health = self.health_check()
        if health.get('status') != 'alive':
            raise ConnectionError(
                f"SAGE daemon not reachable at {self.base_url}. "
                f"Start it with: sudo systemctl start sage-daemon-sprout"
            )
        print(f"DaemonIRP connected to {self.base_url} "
              f"(version={health.get('daemon_version', '?')}, "
              f"machine={health.get('machine', '?')})")

    def health_check(self) -> Dict[str, Any]:
        """Check if the daemon is alive."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/health",
                method='GET'
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def init_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Initialize processing state.

        Args:
            context: Input context with 'prompt' and optional 'memory'

        Returns:
            State dict ready for step()
        """
        prompt = context.get('prompt', '')
        memory = context.get('memory', [])

        return {
            'prompt': prompt,
            'memory': memory,
            'current_response': '',
            'iteration': 0,
            'max_iterations': 1,  # Daemon does its own refinement
            'convergence_threshold': 0.1,
            'energy': 1.0,
            'system_prompt': self.system_prompt,
        }

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        IRP Protocol: Execute one refinement step.

        Sends the prompt to the daemon's /chat endpoint and stores
        the response in the state dict.
        """
        prompt = state['prompt']
        iteration = state.get('iteration', 0)

        # Build conversation context from memory
        metadata = {}
        if state.get('system_prompt'):
            metadata['system_prompt'] = state['system_prompt']
        if state.get('memory'):
            metadata['memory'] = state['memory']

        # POST to daemon /chat
        try:
            payload = json.dumps({
                'message': prompt,
                'sender': self.sender,
                'conversation_id': self.conversation_id,
                'max_wait_seconds': self.max_wait,
                'metadata': metadata,
            }).encode('utf-8')

            req = urllib.request.Request(
                f"{self.base_url}/chat",
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )

            with urllib.request.urlopen(req, timeout=self.max_wait + 10) as resp:
                result = json.loads(resp.read().decode())

            response = result.get('response', '')
            if result.get('error'):
                response = f"[Daemon error: {result['error']}]"

            state['current_response'] = response
            state['energy'] = 0.0  # Converged (daemon handled refinement)
            state['iteration'] = iteration + 1
            state['metabolic_state'] = result.get('metabolic_state', 'unknown')
            state['atp_remaining'] = result.get('atp_remaining', 0)

        except urllib.error.URLError as e:
            state['current_response'] = f"[Daemon unreachable: {e}]"
            state['energy'] = 0.0
            state['iteration'] = iteration + 1

        except Exception as e:
            state['current_response'] = f"[DaemonIRP error: {e}]"
            state['energy'] = 0.0
            state['iteration'] = iteration + 1

        return state

    def halt(self, state: Dict[str, Any]) -> bool:
        """IRP Protocol: Always halt after one step (daemon handles refinement)."""
        return state.get('iteration', 0) >= 1

    def get_response(self, state: Dict[str, Any]) -> str:
        """Extract final response from state."""
        return state.get('current_response', '')

    def energy(self, state: Dict[str, Any]) -> float:
        """IRP Protocol: Return current energy level."""
        return state.get('energy', 1.0)


if __name__ == '__main__':
    # Quick test
    irp = DaemonIRP({
        'daemon_host': 'localhost',
        'daemon_port': 8750,
    })

    print("\n--- Health Check ---")
    print(json.dumps(irp.health_check(), indent=2))

    print("\n--- Chat Test ---")
    state = irp.init_state({'prompt': 'Hello SAGE, who are you?', 'memory': []})
    state = irp.step(state)
    print(f"Response: {state['current_response']}")
    print(f"Energy: {state['energy']}")
    print(f"Halt: {irp.halt(state)}")
