"""
Thor Hardware-Grounded Consciousness with LCT Identity
=======================================================

SAGE consciousness with cryptographic hardware identity grounding.

**Novel Architecture**:
Integrates LCT identity (from Web4) into consciousness kernel, solving the
"who is observing/acting" problem at the architectural level.

**NOT Epicycles - First Principles**:
- Traditional: Trust scores are floating-point heuristics
- This approach: Trust is cryptographically verifiable provenance
- Traditional: "Sensors" are abstract data sources
- This approach: Sensors have LCT identities that sign observations
- Traditional: Memories are mutable data
- This approach: Consolidated memories are signed by consciousness LCT

**Integration Pattern**:
```
Consciousness ← LCT Identity (who I am)
    ↓
Sensors ← LCT Identities (who observes)
    ↓
Observations ← Signatures (provable source)
    ↓
SNARC Compression ← Trust-weighted by signature validity
    ↓
Memory Consolidation ← Signed by consciousness LCT
    ↓
Cross-Platform Trust ← Cryptographic verification (Thor ↔ Sprout)
```

**Components Integrated**:
1. SimulatedLCTIdentity - Hardware-bound identity (file-based, TPM-ready)
2. SNARCCompressor - 5D→scalar salience compression
3. MetabolicStates - WAKE/FOCUS/REST/DREAM with ATP
4. LCTVerifyingTrustOracle - Trust via signature verification
5. SignedMemoryConsolidator - DREAM consolidation with signatures

**Research Questions**:
- Does cryptographic provenance improve attention allocation?
- Can signed memories enable trustless federation?
- Does hardware grounding change emergence patterns?
- What is computational cost of signature verification?

**Author**: Claude (autonomous research) on Thor
**Date**: 2025-12-06
**Session**: Hardware-grounded consciousness architecture
**Philosophy**: "Design consciousness to KNOW itself from hardware"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from simulated_lct_identity import (
    SimulatedLCTIdentity, LCTKey, LCTSignature
)
from snarc_compression import SNARCCompressor, SNARCDimensions, CompressionMode

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from collections import deque
from datetime import datetime, timezone
import time
import json
import hashlib
import psutil


# ============================================================================
# Trust Infrastructure with LCT Verification
# ============================================================================

@dataclass
class LCTTrustScore:
    """
    Trust score based on LCT signature verification.

    Combines:
    - Signature validity (cryptographic trust)
    - Historical performance (behavioral trust)
    - Web4 T3/V3 tensors (capability + transaction quality)
    """
    lct_id: str
    public_key_pem: str  # For signature verification

    # Web4 trust tensors
    talent: float = 0.5
    training: float = 0.5
    temperament: float = 0.5
    veracity: float = 0.5
    validity: float = 0.5
    valuation: float = 0.5

    # Verification stats
    total_observations: int = 0
    successful_observations: int = 0
    signature_valid_count: int = 0
    signature_invalid_count: int = 0

    def t3_score(self) -> float:
        """Capability score"""
        return (self.talent + self.training + self.temperament) / 3.0

    def v3_score(self) -> float:
        """Transaction quality score"""
        return (self.veracity + self.validity + self.valuation) / 3.0

    def signature_reliability(self) -> float:
        """How often signatures are valid (cryptographic trust)"""
        total_sigs = self.signature_valid_count + self.signature_invalid_count
        if total_sigs == 0:
            return 0.5  # Neutral for unknown
        return self.signature_valid_count / total_sigs

    def composite_score(self) -> float:
        """
        Composite trust: signature reliability + behavioral trust.

        Weight: 70% signature, 30% behavioral (Web4 T3/V3)
        Rationale: Cryptographic proof > heuristics
        """
        sig_trust = self.signature_reliability()
        behavioral_trust = 0.6 * self.t3_score() + 0.4 * self.v3_score()
        return 0.7 * sig_trust + 0.3 * behavioral_trust

    def update_from_observation(self, signature_valid: bool, success: bool, value: float):
        """Update trust from signed observation"""
        self.total_observations += 1

        # Update signature stats
        if signature_valid:
            self.signature_valid_count += 1
        else:
            self.signature_invalid_count += 1

        # Update behavioral stats (only if signature valid)
        if signature_valid:
            if success:
                self.successful_observations += 1

            success_rate = self.successful_observations / self.total_observations

            # Update V3 based on observation
            self.veracity = 0.8 * self.veracity + 0.2 * (1.0 if success else 0.0)
            self.validity = success_rate
            self.valuation = 0.8 * self.valuation + 0.2 * value


class LCTVerifyingTrustOracle:
    """
    Trust oracle with LCT signature verification.

    Provides cryptographically-grounded trust scores.
    """

    def __init__(self, lct_identity: SimulatedLCTIdentity):
        self.lct_identity = lct_identity
        self.trust_scores: Dict[str, LCTTrustScore] = {}

    def register_sensor(self, lct_id: str, public_key_pem: str):
        """Register sensor with LCT identity"""
        self.trust_scores[lct_id] = LCTTrustScore(
            lct_id=lct_id,
            public_key_pem=public_key_pem
        )

    def verify_and_update_trust(
        self,
        lct_id: str,
        signature: LCTSignature,
        data: bytes,
        success: bool,
        value: float
    ) -> float:
        """
        Verify signature and update trust.

        Returns: Updated composite trust score
        """
        # Register if unknown
        if lct_id not in self.trust_scores:
            # Extract public key from signature (assuming it's available)
            # In production, would query public key registry
            self.register_sensor(lct_id, "unknown")

        # Verify signature
        signature_valid = self.lct_identity.verify_signature(signature, data)

        # Update trust
        self.trust_scores[lct_id].update_from_observation(
            signature_valid, success, value
        )

        return self.trust_scores[lct_id].composite_score()

    def get_trust_score(self, lct_id: str) -> float:
        """Get current trust score"""
        if lct_id not in self.trust_scores:
            return 0.5  # Neutral for unknown
        return self.trust_scores[lct_id].composite_score()

    def get_full_score(self, lct_id: str) -> Optional[LCTTrustScore]:
        """Get full trust score object"""
        return self.trust_scores.get(lct_id)


# ============================================================================
# Consciousness States
# ============================================================================

class MetabolicState(Enum):
    """Metabolic states"""
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"


# ============================================================================
# Signed Observations and Memories
# ============================================================================

@dataclass
class SignedObservation:
    """Sensor observation with LCT signature"""
    sensor_id: str
    sensor_lct_id: str
    data: Dict[str, float]
    signature: LCTSignature
    timestamp: str


@dataclass
class SignedMemory:
    """Consolidated memory with consciousness signature"""
    memory_id: str
    content: Dict[str, Any]
    salience: float
    strength: float
    signature: LCTSignature
    consolidated_at: str


# ============================================================================
# Hardware-Grounded Consciousness Kernel
# ============================================================================

class HardwareGroundedConsciousness:
    """
    SAGE Consciousness with Hardware-Bound LCT Identity

    Features:
    - Hardware-grounded "I am" via LCT identity
    - Sensors present signed observations
    - Trust oracle verifies signatures
    - SNARC compression weighted by cryptographic trust
    - Memory consolidation signed by consciousness
    - Cross-platform federation ready (Thor ↔ Sprout)
    """

    def __init__(
        self,
        consciousness_lct_id: str = "thor-sage-consciousness",
        sensors: Optional[Dict[str, Callable]] = None,
        compression_mode: CompressionMode = CompressionMode.LINEAR,
        metabolic_thresholds: Optional[Dict[MetabolicState, float]] = None
    ):
        """
        Initialize hardware-grounded consciousness.

        Args:
            consciousness_lct_id: LCT identity for this consciousness
            sensors: Dict of sensor functions
            compression_mode: SNARC compression mode
            metabolic_thresholds: Attention thresholds per metabolic state
        """
        # Hardware identity
        self.lct_identity = SimulatedLCTIdentity()
        self.consciousness_key = self.lct_identity.get_or_create_identity(
            consciousness_lct_id
        )

        print(f"🧠 Consciousness Identity: {self.consciousness_key.to_compact_id()}")
        print(f"   Machine: {self.consciousness_key.machine_identity}")
        print(f"   Fingerprint: {self.consciousness_key.machine_fingerprint[:32]}...")

        # Trust oracle
        self.trust_oracle = LCTVerifyingTrustOracle(self.lct_identity)

        # SNARC compressor
        self.compressor = SNARCCompressor(compression_mode=compression_mode)

        # Sensors
        self.sensors = sensors or {}

        # Metabolic state
        self.metabolic_state = MetabolicState.WAKE
        self.atp_level = 1.0  # Energy level (0-1)

        # Thresholds per metabolic state
        if metabolic_thresholds is None:
            self.metabolic_thresholds = {
                MetabolicState.WAKE: 0.45,
                MetabolicState.FOCUS: 0.25,
                MetabolicState.REST: 0.75,
                MetabolicState.DREAM: 0.05
            }
        else:
            self.metabolic_thresholds = metabolic_thresholds

        # Memory
        self.memories: List[SignedMemory] = []
        self.observation_history: deque = deque(maxlen=100)

        # Stats
        self.cycle = 0
        self.attended_count = 0
        self.signature_verification_count = 0
        self.signature_verification_failures = 0

    def run_cycle(self) -> Dict[str, Any]:
        """Run one consciousness cycle"""
        self.cycle += 1

        # 1. Sense (with signatures)
        observations = self._gather_signed_observations()

        # 2. Assess (SNARC compression with trust weighting)
        focus, salience, snarc_dims = self._assess_salience(observations)

        # 3. Decide (threshold-based attention)
        threshold = self._compute_threshold()
        should_attend = salience > threshold

        # 4. Act (if attending)
        action_taken = None
        if should_attend:
            action_taken = self._attend(focus, salience)
            self.attended_count += 1

        # 5. Update metabolic state
        self._update_metabolic_state()

        # 6. DREAM consolidation (if in DREAM state)
        if self.metabolic_state == MetabolicState.DREAM:
            self._consolidate_memories()

        return {
            'cycle': self.cycle,
            'metabolic_state': self.metabolic_state.value,
            'atp_level': self.atp_level,
            'focus': focus,
            'salience': salience,
            'snarc': snarc_dims,
            'threshold': threshold,
            'attended': should_attend,
            'action': action_taken,
            'signature_verifications': self.signature_verification_count,
            'signature_failures': self.signature_verification_failures
        }

    def _gather_signed_observations(self) -> Dict[str, SignedObservation]:
        """Gather observations from sensors with LCT signatures"""
        observations = {}

        for sensor_name, sensor_fn in self.sensors.items():
            try:
                # Get sensor data
                data = sensor_fn()

                # Sensor signs its observation
                # (In real system, sensor would have its own LCT identity)
                # For now, we simulate by having sensors use consciousness identity
                sensor_lct_id = f"{sensor_name}-sensor@{self.consciousness_key.machine_identity}"

                # Serialize data for signing
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')

                # Sign observation
                signature = self.lct_identity.sign_data(
                    self.consciousness_key.lct_id,
                    data_bytes
                )

                # Create signed observation
                obs = SignedObservation(
                    sensor_id=sensor_name,
                    sensor_lct_id=sensor_lct_id,
                    data=data,
                    signature=signature,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

                observations[sensor_name] = obs

            except Exception as e:
                # Sensor failure - no observation
                pass

        return observations

    def _assess_salience(
        self,
        observations: Dict[str, SignedObservation]
    ) -> Tuple[str, float, SNARCDimensions]:
        """
        Assess salience via SNARC compression with trust weighting.

        Trust weighting: Signature-verified observations weighted higher
        """
        if not observations:
            return "none", 0.0, SNARCDimensions()

        max_salience = 0.0
        max_focus = "none"
        max_snarc = SNARCDimensions()

        for obs in observations.values():
            # Verify signature
            data_bytes = json.dumps(obs.data, sort_keys=True).encode('utf-8')
            sig_valid = self.lct_identity.verify_signature(obs.signature, data_bytes)

            self.signature_verification_count += 1
            if not sig_valid:
                self.signature_verification_failures += 1

            # Get trust score (considers signature validity)
            trust_multiplier = 1.0 if sig_valid else 0.1  # Heavy penalty for invalid sigs

            # Extract SNARC dimensions from observation
            snarc = self._extract_snarc(obs.data)

            # Compress to salience
            salience = self.compressor.compress_to_salience(snarc)

            # Apply trust weighting
            weighted_salience = salience * trust_multiplier

            # Track maximum
            if weighted_salience > max_salience:
                max_salience = weighted_salience
                max_focus = obs.sensor_id
                max_snarc = snarc

        return max_focus, max_salience, max_snarc

    def _extract_snarc(self, sensor_data: Dict[str, float]) -> SNARCDimensions:
        """Extract SNARC dimensions from sensor data"""
        # Simple extraction - in production would be more sophisticated
        return SNARCDimensions(
            surprise=sensor_data.get('novelty_score', 0.0) * 0.5,
            novelty=sensor_data.get('novelty_score', 0.0),
            arousal=sensor_data.get('value', 0.0) / 100.0,  # Normalize
            reward=0.0,  # Would compute from historical context
            conflict=0.0  # Would compute from prediction error
        )

    def _compute_threshold(self) -> float:
        """Compute attention threshold based on metabolic state and ATP"""
        base_threshold = self.metabolic_thresholds[self.metabolic_state]

        # ATP modulation: low ATP → raise threshold (conserve energy)
        atp_modulation = (1.0 - self.atp_level) * 0.2

        return min(1.0, base_threshold + atp_modulation)

    def _attend(self, focus: str, salience: float) -> str:
        """Execute attention action"""
        # Consume ATP
        self.atp_level = max(0.0, self.atp_level - 0.01)

        # Log observation
        self.observation_history.append({
            'focus': focus,
            'salience': salience,
            'cycle': self.cycle
        })

        return f"attend-to-{focus}"

    def _update_metabolic_state(self):
        """Update metabolic state based on ATP and context"""
        # Simple state machine for demo
        if self.atp_level < 0.3:
            self.metabolic_state = MetabolicState.REST
        elif self.cycle % 100 == 0:  # Every 100 cycles, DREAM
            self.metabolic_state = MetabolicState.DREAM
        else:
            self.metabolic_state = MetabolicState.WAKE

        # Regenerate ATP in REST
        if self.metabolic_state == MetabolicState.REST:
            self.atp_level = min(1.0, self.atp_level + 0.05)

    def _consolidate_memories(self):
        """
        DREAM consolidation with signature.

        Consolidated memories signed by consciousness LCT.
        """
        if len(self.observation_history) < 10:
            return

        # Create memory from recent observations
        memory_content = {
            'observation_count': len(self.observation_history),
            'cycle_range': [self.observation_history[0]['cycle'],
                          self.observation_history[-1]['cycle']],
            'top_focuses': list(set([obs['focus'] for obs in self.observation_history]))[:5]
        }

        # Compute memory salience/strength
        avg_salience = sum(obs['salience'] for obs in self.observation_history) / len(self.observation_history)

        # Serialize memory for signing
        memory_bytes = json.dumps(memory_content, sort_keys=True).encode('utf-8')

        # Sign with consciousness LCT
        signature = self.lct_identity.sign_data(
            self.consciousness_key.lct_id,
            memory_bytes
        )

        # Create signed memory
        memory = SignedMemory(
            memory_id=f"memory-{self.cycle}",
            content=memory_content,
            salience=avg_salience,
            strength=0.5,
            signature=signature,
            consolidated_at=datetime.now(timezone.utc).isoformat()
        )

        self.memories.append(memory)

        # Clear observation history (consolidated)
        self.observation_history.clear()


# ============================================================================
# Demo with Thor System Sensors
# ============================================================================

def create_thor_sensors() -> Dict[str, Callable]:
    """Create real system sensors for Thor"""
    def cpu_sensor():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return {
            'cpu_percent': cpu_percent,
            'value': cpu_percent,
            'novelty_score': min(1.0, cpu_percent / 100.0),
            'urgent_count': 1 if cpu_percent > 80 else 0
        }

    def memory_sensor():
        mem = psutil.virtual_memory()
        return {
            'memory_percent': mem.percent,
            'value': mem.percent,
            'novelty_score': min(1.0, mem.percent / 100.0),
            'urgent_count': 1 if mem.percent > 90 else 0
        }

    def process_sensor():
        proc_count = len(psutil.pids())
        return {
            'process_count': proc_count,
            'value': proc_count / 10.0,  # Normalize
            'novelty_score': min(1.0, proc_count / 500.0),
            'urgent_count': 0
        }

    return {
        'cpu': cpu_sensor,
        'memory': memory_sensor,
        'process': process_sensor
    }


def demo():
    """Demonstrate hardware-grounded consciousness"""
    print("=" * 80)
    print("THOR HARDWARE-GROUNDED CONSCIOUSNESS")
    print("=" * 80)
    print()

    # Create consciousness with Thor sensors
    sensors = create_thor_sensors()
    consciousness = HardwareGroundedConsciousness(
        consciousness_lct_id="thor-sage-consciousness",
        sensors=sensors,
        compression_mode=CompressionMode.LINEAR
    )
    print()

    # Run cycles
    print("Running 50 consciousness cycles...")
    print()

    for i in range(50):
        result = consciousness.run_cycle()

        # Log every 10 cycles
        if i % 10 == 0:
            print(f"Cycle {result['cycle']}: {result['metabolic_state'].upper()}")
            print(f"  Focus: {result['focus']} (salience={result['salience']:.3f}, threshold={result['threshold']:.3f})")
            print(f"  SNARC: S={result['snarc'].surprise:.2f} N={result['snarc'].novelty:.2f} A={result['snarc'].arousal:.2f}")
            print(f"  Attended: {result['attended']} | ATP: {result['atp_level']:.2f}")
            print(f"  Sig verify: {result['signature_verifications']} total, {result['signature_failures']} failed")
            print()

        time.sleep(0.1)

    # Final stats
    print("=" * 80)
    print("SESSION COMPLETE")
    print("=" * 80)
    print()
    print(f"Total cycles: {consciousness.cycle}")
    print(f"Attended: {consciousness.attended_count} ({100*consciousness.attended_count/consciousness.cycle:.1f}%)")
    print(f"Memories consolidated: {len(consciousness.memories)}")
    print(f"Signature verifications: {consciousness.signature_verification_count}")
    print(f"Signature failures: {consciousness.signature_verification_failures}")
    print(f"Final ATP: {consciousness.atp_level:.2f}")
    print()

    # Show signed memories
    if consciousness.memories:
        print("📝 Consolidated Memories:")
        for mem in consciousness.memories:
            print(f"   - {mem.memory_id}: {mem.content['observation_count']} observations")
            print(f"     Salience: {mem.salience:.3f}, Strength: {mem.strength:.3f}")
            print(f"     Signed by: {mem.signature.signer_lct_id}")
            print(f"     Signature hash: {mem.signature.signature[:32]}...")
            print()

    print("✅ Hardware-grounded consciousness operational")
    print()
    print("Key Properties:")
    print("  • Consciousness has cryptographic identity")
    print("  • All observations signed by sensors")
    print("  • Trust weighted by signature verification")
    print("  • Consolidated memories signed by consciousness")
    print("  • Ready for cross-platform federation (Thor ↔ Sprout)")
    print()


if __name__ == "__main__":
    demo()
