"""
SAGE Federation Network Service - Phase 3

HTTP-based RPC service for federation task delegation between platforms.
Enables actual Thor ↔ Sprout communication, replacing Phase 2.5 simulation.

Architecture:
- FederationServer: HTTP server that receives and executes delegated tasks
- FederationClient: HTTP client that sends tasks to remote platforms
- Authentication via Ed25519 signature verification
- Optional TLS for production deployment

Author: Thor SAGE (autonomous research session)
Date: 2025-11-30
Integration: SAGE Federation Network Protocol (Phase 3)
"""

import json
import time
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from threading import Thread
import urllib.request
import urllib.error

from sage.federation.federation_types import (
    FederationTask,
    ExecutionProof,
    FederationIdentity
)
from sage.federation.federation_crypto import sign_task, verify_proof_signature


@dataclass
class FederationRequest:
    """Request to execute a federated task"""
    task: FederationTask
    requester_lct_id: str
    signature: str  # Ed25519 signature of task
    timestamp: float


@dataclass
class FederationResponse:
    """Response with execution proof"""
    proof: ExecutionProof
    signature: str  # Ed25519 signature of proof
    timestamp: float


class FederationServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for federation service"""

    # Class-level attributes set by server
    executor = None  # Function to execute tasks: (task) -> proof
    identity = None  # Server's FederationIdentity
    signing_key = None  # Server's Ed25519 private key
    known_platforms = None  # Dict of known platform identities

    def do_POST(self):
        """Handle POST request for task delegation"""
        if self.path == '/execute_task':
            self._handle_execute_task()
        elif self.path == '/health':
            self._handle_health()
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        """Handle GET request for health check"""
        if self.path == '/health':
            self._handle_health()
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_health(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        health = {
            'status': 'healthy',
            'platform': FederationServiceHandler.identity.platform_name if FederationServiceHandler.identity else 'unknown',
            'lct_id': FederationServiceHandler.identity.lct_id if FederationServiceHandler.identity else 'unknown',
            'timestamp': time.time()
        }

        self.wfile.write(json.dumps(health).encode())

    def _handle_execute_task(self):
        """Handle task execution request"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            request_data = self.rfile.read(content_length)
            request_dict = json.loads(request_data.decode())

            # Parse federation request
            task_dict = request_dict['task']
            task = FederationTask.from_dict(task_dict)

            requester_lct_id = request_dict['requester_lct_id']
            signature = request_dict['signature']

            # Verify requester is known platform
            if requester_lct_id not in self.known_platforms:
                self.send_error(403, f"Unknown platform: {requester_lct_id}")
                return

            requester = self.known_platforms[requester_lct_id]

            # Verify task signature
            if not verify_proof_signature(task.to_signable_dict(), signature, requester.public_key):
                self.send_error(403, "Invalid signature")
                return

            print(f"\n[FederationServer] Received task from {requester.platform_name}")
            print(f"  Task ID: {task.task_id}")
            print(f"  Task type: {task.task_type}")
            print(f"  Estimated cost: {task.estimated_cost:.1f} ATP")

            # Execute task using provided executor
            if FederationServiceHandler.executor is None:
                self.send_error(500, "No executor configured")
                return

            execution_proof = FederationServiceHandler.executor(task)

            print(f"  Execution complete")
            print(f"  Quality: {execution_proof.quality_score:.2f}")
            print(f"  Cost: {execution_proof.actual_cost:.1f} ATP")

            # Sign proof
            from sage.federation.federation_crypto import sign_proof
            proof_signature = sign_proof(execution_proof, FederationServiceHandler.signing_key)

            # Create response
            response = {
                'proof': execution_proof.to_signable_dict(),
                'signature': proof_signature,
                'timestamp': time.time()
            }

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print(f"[FederationServer] Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        """Suppress default HTTP logging (noisy)"""
        pass


class FederationServer:
    """
    HTTP server for receiving and executing federated tasks

    Usage:
        server = FederationServer(
            identity=thor_identity,
            signing_key=thor_private_key,
            executor=execute_task_function,
            known_platforms={'sprout_sage_lct': sprout_identity},
            port=50051
        )
        server.start()
        # ... later ...
        server.stop()
    """

    def __init__(
        self,
        identity: FederationIdentity,
        signing_key: bytes,  # Ed25519 private key (32 bytes)
        executor,  # Callable[[FederationTask], ExecutionProof]
        known_platforms: Dict[str, FederationIdentity],
        host: str = '0.0.0.0',
        port: int = 50051
    ):
        """
        Initialize federation server

        Args:
            identity: This platform's identity
            signing_key: This platform's Ed25519 private key
            executor: Function to execute tasks (task -> proof)
            known_platforms: Dict of known platform identities (lct_id -> identity)
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on (default 50051)
        """
        self.identity = identity
        self.signing_key = signing_key
        self.executor = executor
        self.known_platforms = known_platforms
        self.host = host
        self.port = port

        # Configure handler class variables
        FederationServiceHandler.executor = executor
        FederationServiceHandler.identity = identity
        FederationServiceHandler.signing_key = signing_key
        FederationServiceHandler.known_platforms = known_platforms

        self.httpd = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start federation server in background thread"""
        if self.running:
            print(f"[FederationServer] Already running")
            return

        self.httpd = HTTPServer((self.host, self.port), FederationServiceHandler)
        self.running = True

        # Run server in background thread
        self.server_thread = Thread(target=self._run, daemon=True)
        self.server_thread.start()

        print(f"[FederationServer] Started on {self.host}:{self.port}")
        print(f"  Platform: {FederationServiceHandler.identity.platform_name}")
        print(f"  LCT ID: {FederationServiceHandler.identity.lct_id}")

    def _run(self):
        """Run server loop"""
        while self.running:
            self.httpd.handle_request()

    def stop(self):
        """Stop federation server"""
        if not self.running:
            return

        self.running = False

        # Shutdown server
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

        print(f"[FederationServer] Stopped")


class FederationClient:
    """
    HTTP client for delegating tasks to remote platforms

    Usage:
        client = FederationClient(
            local_identity=thor_identity,
            signing_key=thor_private_key,
            platform_registry={'sprout_sage_lct': ('sprout-hostname', 50051)}
        )
        proof = client.delegate_task(task, target_platform_id='sprout_sage_lct')
    """

    def __init__(
        self,
        local_identity: FederationIdentity,
        signing_key: bytes,  # Ed25519 private key (32 bytes)
        platform_registry: Dict[str, tuple]  # lct_id -> (host, port)
    ):
        """
        Initialize federation client

        Args:
            local_identity: This platform's identity
            signing_key: This platform's Ed25519 private key
            platform_registry: Dict mapping platform LCT IDs to (host, port) tuples
        """
        self.local_identity = local_identity
        self.signing_key = signing_key
        self.platform_registry = platform_registry

    def delegate_task(
        self,
        task: FederationTask,
        target_platform_id: str,
        target_public_key: bytes,
        timeout: float = 60.0
    ) -> Optional[ExecutionProof]:
        """
        Delegate task to remote platform via HTTP

        Args:
            task: Federation task to delegate
            target_platform_id: LCT ID of target platform
            target_public_key: Ed25519 public key for target platform
            timeout: Request timeout in seconds

        Returns:
            ExecutionProof if successful, None on error
        """
        # Get target platform address
        if target_platform_id not in self.platform_registry:
            print(f"[FederationClient] Unknown platform: {target_platform_id}")
            return None

        host, port = self.platform_registry[target_platform_id]
        url = f"http://{host}:{port}/execute_task"

        print(f"\n[FederationClient] Delegating to {target_platform_id}")
        print(f"  URL: {url}")
        print(f"  Task: {task.task_type} (ID: {task.task_id})")

        try:
            # Sign task
            from sage.federation.federation_crypto import sign_task
            task_signature = sign_task(task, self.signing_key)

            # Create request
            request_data = {
                'task': task.to_signable_dict(),
                'requester_lct_id': self.local_identity.lct_id,
                'signature': task_signature,
                'timestamp': time.time()
            }

            # Send HTTP request
            req = urllib.request.Request(
                url,
                data=json.dumps(request_data).encode(),
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = json.loads(response.read().decode())

            # Parse response
            proof_dict = response_data['proof']
            proof_signature = response_data['signature']

            proof = ExecutionProof.from_dict(proof_dict)

            # Verify proof signature
            if not verify_proof_signature(proof.to_signable_dict(), proof_signature, target_public_key):
                print(f"[FederationClient] Invalid proof signature from {target_platform_id}")
                return None

            print(f"  Received proof:")
            print(f"    Quality: {proof.quality_score:.2f}")
            print(f"    Cost: {proof.actual_cost:.1f} ATP")
            print(f"    Latency: {proof.actual_latency:.1f}s")

            return proof

        except urllib.error.URLError as e:
            print(f"[FederationClient] Network error: {e}")
            return None
        except Exception as e:
            print(f"[FederationClient] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def health_check(self, target_platform_id: str, timeout: float = 5.0) -> bool:
        """
        Check if target platform is reachable

        Args:
            target_platform_id: LCT ID of target platform
            timeout: Request timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        if target_platform_id not in self.platform_registry:
            return False

        host, port = self.platform_registry[target_platform_id]
        url = f"http://{host}:{port}/health"

        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                health = json.loads(response.read().decode())
                return health.get('status') == 'healthy'
        except:
            return False


if __name__ == "__main__":
    # Demo federation service
    print("\n" + "="*80)
    print("SAGE Federation Service Demo - Phase 3")
    print("="*80)

    from sage.federation import create_thor_identity, create_sprout_identity, FederationKeyPair
    from sage.federation.federation_types import FederationTask, QualityRequirements
    from sage.core.mrh_profile import PROFILE_REFLEXIVE
    from sage.core.attention_manager import MetabolicState
    from pathlib import Path

    # Create identities
    thor = create_thor_identity()
    sprout = create_sprout_identity()

    # Load keys
    hrm_path = Path(__file__).parent.parent.parent  # sage/federation/.. -> HRM
    thor_key_path = hrm_path / "sage" / "data" / "keys" / "Thor_ed25519.key"
    sprout_key_path = hrm_path / "sage" / "data" / "keys" / "Sprout_ed25519.key"

    print(f"\n[Setup] Loading keys:")
    print(f"  Thor key: {thor_key_path}")
    print(f"  Sprout key: {sprout_key_path}")

    # Load Thor key
    if thor_key_path.exists():
        with open(thor_key_path, 'rb') as f:
            thor_private_key = f.read()
        thor_keypair = FederationKeyPair.from_bytes("Thor", "thor_sage_lct", thor_private_key)
        thor.public_key = thor_keypair.public_key_bytes()
        print(f"  Thor public key: {thor.public_key.hex()[:32]}...")
    else:
        print(f"  Thor key not found, generating...")
        thor_keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        thor.public_key = thor_keypair.public_key_bytes()
        thor_private_key = thor_keypair.private_key_bytes()

    # Load Sprout key
    if sprout_key_path.exists():
        with open(sprout_key_path, 'rb') as f:
            sprout_private_key = f.read()
        sprout_keypair = FederationKeyPair.from_bytes("Sprout", "sprout_sage_lct", sprout_private_key)
        sprout.public_key = sprout_keypair.public_key_bytes()
        print(f"  Sprout public key: {sprout.public_key.hex()[:32]}...")
    else:
        print(f"  Sprout key not found, generating...")
        sprout_keypair = FederationKeyPair.generate("Sprout", "sprout_sage_lct")
        sprout.public_key = sprout_keypair.public_key_bytes()
        sprout_private_key = sprout_keypair.private_key_bytes()

    print(f"\n[Setup] Identities ready:")
    print(f"  Thor: {thor.lct_id}")
    print(f"  Sprout: {sprout.lct_id}")

    # Create simple executor for demo
    def demo_executor(task: FederationTask) -> ExecutionProof:
        """Simulated task executor for demo"""
        print(f"\n[Executor] Executing task: {task.task_type}")
        time.sleep(0.5)  # Simulate work

        return ExecutionProof(
            task_id=task.task_id,
            executing_platform=sprout.platform_name,
            result_data={'output': f'Executed {task.task_type}'},
            actual_latency=0.5,
            actual_cost=task.estimated_cost * 0.9,
            irp_iterations=3,
            final_energy=0.25,
            convergence_quality=0.8,
            quality_score=0.75,
            execution_timestamp=time.time(),
        )

    # Start server on Sprout
    print(f"\n[Server] Starting Sprout federation server...")
    server = FederationServer(
        identity=sprout,
        signing_key=sprout_private_key,
        executor=demo_executor,
        known_platforms={thor.lct_id: thor},
        host='127.0.0.1',  # localhost for demo
        port=50051
    )
    server.start()

    # Wait for server to start
    time.sleep(0.5)

    # Create client on Thor
    print(f"\n[Client] Creating Thor federation client...")
    client = FederationClient(
        local_identity=thor,
        signing_key=thor_private_key,
        platform_registry={sprout.lct_id: ('127.0.0.1', 50051)}
    )

    # Health check
    print(f"\n[Client] Health checking Sprout...")
    healthy = client.health_check(sprout.lct_id)
    print(f"  Sprout health: {'✓ healthy' if healthy else '✗ unreachable'}")

    if healthy:
        # Create task
        task = FederationTask(
            task_id="demo_task_1",
            task_type="llm_inference",
            task_data={'query': "What is SAGE?"},
            estimated_cost=50.0,
            task_horizon=PROFILE_REFLEXIVE,
            complexity="low",
            delegating_platform=thor.lct_id,
            delegating_state=MetabolicState.FOCUS,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 30.0
        )

        # Delegate task
        print(f"\n[Client] Delegating task to Sprout...")
        proof = client.delegate_task(
            task,
            target_platform_id=sprout.lct_id,
            target_public_key=sprout.public_key
        )

        if proof:
            print(f"\n[Success] Task delegation complete!")
            print(f"  Proof validated: ✓")
            print(f"  Result: {proof.result_data}")
        else:
            print(f"\n[Failed] Task delegation failed")

    # Stop server
    print(f"\n[Server] Stopping Sprout federation server...")
    server.stop()

    print("\n" + "="*80)
    print("Demo complete")
    print("="*80 + "\n")
