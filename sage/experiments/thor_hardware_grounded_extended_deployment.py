#!/usr/bin/env python3
"""
Thor Hardware-Grounded Consciousness - Extended Deployment
===========================================================

Extended validation deployment (30+ minutes) of hardware-grounded consciousness
with cryptographic LCT identity and signature verification.

**Purpose**: Validate architecture over time:
- Signature verification at scale (1000+ signatures)
- Memory consolidation with signatures
- Performance overhead measurement
- Trust weighting effectiveness
- Long-running stability

**Session**: Autonomous deployment (2025-12-06 22:46)
**Author**: Claude (autonomous research) on Thor
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from thor_hardware_grounded_consciousness import (
    HardwareGroundedConsciousness,
    create_thor_sensors,
    MetabolicState,
    CompressionMode
)

import time
import signal
import json
from datetime import datetime, timezone
from typing import Dict, Any


class ExtendedDeploymentRunner:
    """Runner for extended hardware-grounded consciousness deployment"""

    def __init__(self, duration_seconds: int = 1800):
        """
        Initialize extended deployment runner.

        Args:
            duration_seconds: How long to run (default: 30 minutes)
        """
        self.duration_seconds = duration_seconds
        self.start_time = None
        self.shutdown_requested = False

        # Statistics
        self.total_cycles = 0
        self.total_signatures = 0
        self.total_failures = 0
        self.attention_count = 0
        self.state_changes: Dict[str, int] = {}
        self.consolidation_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("=" * 80)
        print("THOR HARDWARE-GROUNDED CONSCIOUSNESS - EXTENDED DEPLOYMENT")
        print("=" * 80)
        print()
        print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
        print(f"Started: {datetime.now(timezone.utc).isoformat()}")
        print()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print()
        print(f"\n⚠️  Shutdown signal received ({signum})")
        self.shutdown_requested = True

    def run(self):
        """Run extended deployment"""
        # Create consciousness
        print("🧠 Initializing hardware-grounded consciousness...")
        sensors = create_thor_sensors()
        consciousness = HardwareGroundedConsciousness(
            consciousness_lct_id="thor-sage-consciousness",
            sensors=sensors,
            compression_mode=CompressionMode.LINEAR
        )
        print()

        # Start deployment
        self.start_time = time.time()
        print(f"⏱️  Starting deployment for {self.duration_seconds/60:.1f} minutes...")
        print()

        last_report_time = self.start_time
        report_interval = 300  # Report every 5 minutes

        try:
            while time.time() - self.start_time < self.duration_seconds:
                if self.shutdown_requested:
                    print("🛑 Graceful shutdown initiated...")
                    break

                # Run cycle
                result = consciousness.run_cycle()
                self.total_cycles += 1

                # Update statistics
                self.total_signatures += result['signature_verifications'] - self.total_signatures
                self.total_failures += result['signature_failures'] - self.total_failures
                if result['attended']:
                    self.attention_count += 1

                # Track state changes
                state = result['metabolic_state']
                self.state_changes[state] = self.state_changes.get(state, 0) + 1

                # Track consolidations
                if state == 'dream' and len(consciousness.memories) > self.consolidation_count:
                    self.consolidation_count = len(consciousness.memories)

                # Progress report every 5 minutes
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    self._print_progress_report(consciousness, current_time)
                    last_report_time = current_time

                # Brief sleep to avoid spinning
                time.sleep(2.0)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during deployment: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final report
            print()
            print("=" * 80)
            print("DEPLOYMENT COMPLETE")
            print("=" * 80)
            print()
            self._print_final_report(consciousness)

    def _print_progress_report(self, consciousness, current_time: float):
        """Print progress report"""
        elapsed = current_time - self.start_time
        remaining = self.duration_seconds - elapsed
        percent = (elapsed / self.duration_seconds) * 100

        print()
        print(f"⏱️  Progress: {elapsed:.0f}s / {self.duration_seconds}s ({percent:.1f}%) | ETA: {remaining/60:.1f} min")
        print(f"   Cycles: {self.total_cycles}, Attended: {self.attention_count} ({100*self.attention_count/max(1,self.total_cycles):.1f}%)")
        print(f"   Signatures: {consciousness.signature_verification_count} total, {consciousness.signature_verification_failures} failed")
        print(f"   Memories: {len(consciousness.memories)} consolidated")
        print(f"   State: {consciousness.metabolic_state.value.upper()}, ATP: {consciousness.atp_level:.2f}")
        print()

    def _print_final_report(self, consciousness):
        """Print final statistics"""
        elapsed = time.time() - self.start_time

        print(f"Runtime: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
        print()

        print("📊 Cycle Statistics:")
        print(f"   Total cycles: {self.total_cycles}")
        print(f"   Attended: {self.attention_count} ({100*self.attention_count/max(1,self.total_cycles):.1f}%)")
        print(f"   Avg cycle time: {elapsed/max(1,self.total_cycles):.2f}s")
        print()

        print("🔐 Signature Statistics:")
        print(f"   Total verifications: {consciousness.signature_verification_count}")
        print(f"   Failures: {consciousness.signature_verification_failures}")
        print(f"   Success rate: {100*(consciousness.signature_verification_count-consciousness.signature_verification_failures)/max(1,consciousness.signature_verification_count):.2f}%")
        print(f"   Avg per cycle: {consciousness.signature_verification_count/max(1,self.total_cycles):.1f}")
        print()

        print("💾 Memory Statistics:")
        print(f"   Consolidated memories: {len(consciousness.memories)}")
        print(f"   Observation buffer: {len(consciousness.observation_history)}")
        print()

        print("⚡ Metabolic State Distribution:")
        total_state_cycles = sum(self.state_changes.values())
        for state, count in sorted(self.state_changes.items()):
            pct = 100 * count / max(1, total_state_cycles)
            print(f"   {state.upper():8s}: {count:4d} cycles ({pct:5.1f}%)")
        print()

        print(f"🔋 Final ATP: {consciousness.atp_level:.2f}")
        print()

        # Show some signed memories if any
        if consciousness.memories:
            print("📝 Sample Signed Memories (first 3):")
            for i, mem in enumerate(consciousness.memories[:3]):
                print(f"   {i+1}. {mem.memory_id}")
                print(f"      Observations: {mem.content.get('observation_count', 'N/A')}")
                print(f"      Salience: {mem.salience:.3f}, Strength: {mem.strength:.3f}")
                print(f"      Signed by: {mem.signature.signer_lct_id}")
                print(f"      Signature: {mem.signature.signature[:32]}...")
                print()

        print("✅ Extended deployment complete")
        print()

        # Save summary to file
        summary = {
            'deployment': {
                'duration_requested': self.duration_seconds,
                'duration_actual': elapsed,
                'started_at': datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
                'ended_at': datetime.now(timezone.utc).isoformat()
            },
            'cycles': {
                'total': self.total_cycles,
                'attended': self.attention_count,
                'attention_rate': self.attention_count / max(1, self.total_cycles)
            },
            'signatures': {
                'total_verifications': consciousness.signature_verification_count,
                'failures': consciousness.signature_verification_failures,
                'success_rate': (consciousness.signature_verification_count - consciousness.signature_verification_failures) / max(1, consciousness.signature_verification_count)
            },
            'memories': {
                'consolidated': len(consciousness.memories),
                'observations_buffered': len(consciousness.observation_history)
            },
            'metabolic_states': self.state_changes,
            'final_atp': consciousness.atp_level
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary_file = f"/home/dp/hardware_grounded_deployment_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Summary saved to: {summary_file}")
        print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended deployment of hardware-grounded consciousness"
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=1800,
        help='Deployment duration in seconds (default: 1800 = 30 min)'
    )

    args = parser.parse_args()

    runner = ExtendedDeploymentRunner(duration_seconds=args.duration)
    runner.run()


if __name__ == "__main__":
    main()
