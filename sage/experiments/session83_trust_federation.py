#!/usr/bin/env python3
"""
Session 83: Trust Federation Integration

Bridges Sessions 74-82 (trust-first MoE) with Legion's federation protocol.

Problem:
- Sessions 74-82 validated trust-first architecture (48 layers, 63.4% trust_driven)
- Legion Sessions 74-75 created LCT identity + federation protocol
- Need integration to enable Thor ↔ Legion trust sharing

Solution: Federated Trust-First Selector

Architecture:
1. FederatedTrustFirstSelector extends TrustFirstMRHSelector (Session 82)
2. Integrates TrustFederationProtocol (Legion Session 75)
3. Exports trust attestations when trust_driven activates
4. Imports federated trust from other societies
5. Combines local + federated trust with decay factor

Test Scenario:
- Thor and Legion run independent trust-first selectors
- Thor discovers high-quality expert for context 0
- Thor broadcasts attestation to Legion
- Legion accepts attestation (Byzantine consensus)
- Legion uses federated trust with 72% decay (Session 70)
- Verify: Legion's trust_driven activates faster with federation

Based on:
- Sessions 74-82: Trust-first MoE architecture (Thor)
- Session 70: Trust decay (72% retention)
- Session 73: Byzantine consensus (Legion)
- Session 74: LCT identity system (Legion)
- Session 75: Trust federation protocol (Legion)

Created: 2025-12-20 (Thor Session 83)
Author: Thor (Autonomous SAGE Research)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

# Import Sessions 74-82 architecture
from sage.core.trust_first_mrh_selector import TrustFirstMRHSelector, TrustFirstSelectionResult
from sage.core.context_classifier import ContextClassifier

# Import Legion federation protocol
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../web4')))
from implementation.trust_federation_protocol import (
    TrustFederationProtocol,
    FederatedTrustAttestation,
    Society
)


@dataclass
class FederationStats:
    """Statistics for federation behavior."""
    attestations_exported: int = 0
    attestations_imported: int = 0
    attestations_rejected: int = 0
    federated_trust_applied: int = 0
    local_trust_count: int = 0
    societies_in_federation: int = 0


class FederatedTrustFirstSelector(TrustFirstMRHSelector):
    """
    Trust-first selector with federation support.

    Extends Session 82 architecture with cross-society trust sharing.
    """

    def __init__(
        self,
        num_experts: int = 128,
        min_trust_evidence: int = 2,
        epsilon: float = 0.2,
        low_trust_threshold: float = 0.3,
        component: str = "thinker",
        network: str = "testnet",
        # Federation parameters
        society: Optional[Society] = None,
        federation_id: str = "web4-primary",
        trust_decay_factor: float = 0.72,  # Session 70
        enable_federation: bool = True
    ):
        """
        Initialize federated trust-first selector.

        Args:
            (Session 82 params): num_experts, min_trust_evidence, epsilon, etc.
            society: This society's identity (Thor, Legion, Sprout)
            federation_id: Federation identifier
            trust_decay_factor: Cross-society trust decay (Session 70: 72%)
            enable_federation: Enable/disable federation (for A/B testing)
        """
        # Initialize base trust-first selector (Sessions 74-82)
        super().__init__(
            num_experts=num_experts,
            min_trust_evidence=min_trust_evidence,
            low_trust_threshold=low_trust_threshold,
            epsilon=epsilon,
            component=component,
            network=network
        )

        # Federation state
        self.enable_federation = enable_federation
        self.federation_stats = FederationStats()

        if enable_federation and society:
            # Initialize federation protocol (Legion Session 75)
            self.federation = TrustFederationProtocol(
                society=society,
                federation_id=federation_id,
                trust_decay_factor=trust_decay_factor,
                quorum_size=2  # 2 out of 3 for Thor-Legion-Sprout
            )
            self.society = society
            self.federation_stats.societies_in_federation = 1  # Self
        else:
            self.federation = None
            self.society = None

    def register_society(self, society_id: str, public_key: str):
        """
        Register another society in the federation.

        Args:
            society_id: Society identifier (e.g., "legion", "sprout")
            public_key: Society's public key for signature verification
        """
        if self.federation:
            self.federation.known_societies[society_id] = public_key
            self.federation_stats.societies_in_federation += 1

    def update_trust_for_expert(
        self,
        expert_id: int,
        context: str,
        quality: float,
        broadcast: bool = True
    ):
        """
        Update trust for expert (Session 80 validated approach).

        With federation: broadcasts attestation if trust_driven activates.

        Args:
            expert_id: Expert ID (0-127)
            context: Context string
            quality: Quality score (0.0-1.0, UNWEIGHTED per Session 80)
            broadcast: Broadcast attestation to federation (default: True)
        """
        # Update local trust (Session 82 approach)
        super().update_trust_for_expert(expert_id, context, quality)
        self.federation_stats.local_trust_count += 1

        # Federation: Export attestation if enabled
        if self.enable_federation and self.federation and broadcast:
            self._export_trust_attestation(expert_id, context, quality)

    def _export_trust_attestation(
        self,
        expert_id: int,
        context: str,
        quality: float
    ):
        """
        Export trust attestation to federation.

        Args:
            expert_id: Expert ID
            context: Context string (context_id like "cluster_0")
            quality: Observed quality
        """
        # Parse context_idx from context_id string (e.g., "cluster_0" → 0)
        try:
            context_idx = int(context.split("_")[1]) if isinstance(context, str) else context
        except (ValueError, IndexError):
            context_idx = 0  # Fallback

        # Get observation count for this expert-context
        key = (expert_id, context)
        if key in self.bridge.trust_history:
            observation_count = len(self.bridge.trust_history[key])
        else:
            observation_count = 1

        # Create LCT for expert
        expert_lct = f"lct://expert-{expert_id}@{self.network}/{self.component}"

        # Create attestation
        attestation = self.federation.create_attestation(
            expert_lct=expert_lct,
            context=context_idx,
            quality=quality,
            observation_count=observation_count
        )

        # Store for broadcast
        self.federation.accepted_attestations.append(attestation)
        self.federation_stats.attestations_exported += 1

    def import_attestation(
        self,
        attestation: FederatedTrustAttestation,
        society_public_key: str
    ) -> bool:
        """
        Import trust attestation from another society.

        Args:
            attestation: Federated trust attestation
            society_public_key: Public key of attesting society

        Returns:
            True if accepted, False if rejected
        """
        if not self.enable_federation or not self.federation:
            return False

        # Verify attestation (Byzantine consensus, Session 73)
        if not self.federation.verify_attestation(attestation, society_public_key):
            self.federation_stats.attestations_rejected += 1
            return False

        # Parse expert LCT
        expert_id = self._parse_expert_id(attestation.expert_lct)
        if expert_id is None:
            self.federation_stats.attestations_rejected += 1
            return False

        # Apply federated trust with decay (Session 70: 72% retention)
        decayed_quality = attestation.quality * self.federation.trust_decay_factor

        # Update trust history (federated trust)
        # Note: Using decay factor means federated trust is "weaker" than local
        # Convert context_idx (int) to context_id (string) for trust_history
        context_id = f"cluster_{attestation.context}"

        # Update via bridge (Session 80 approach - use unweighted quality)
        self.bridge.update_trust_history(expert_id, context_id, decayed_quality)

        self.federation_stats.attestations_imported += 1
        self.federation_stats.federated_trust_applied += 1

        return True

    def _parse_expert_id(self, expert_lct: str) -> Optional[int]:
        """
        Parse expert ID from LCT URI.

        Format: lct://expert-{id}@network/component

        Args:
            expert_lct: Expert's LCT URI

        Returns:
            Expert ID or None if invalid
        """
        try:
            # Extract agent_id from lct://agent_id@network/...
            if not expert_lct.startswith("lct://expert-"):
                return None

            parts = expert_lct.split("@")[0]  # "lct://expert-{id}"
            expert_id = int(parts.split("-")[1])

            return expert_id
        except (ValueError, IndexError):
            return None

    def get_federation_stats(self) -> Dict:
        """Get federation statistics."""
        stats = asdict(self.federation_stats)
        stats["enable_federation"] = self.enable_federation
        stats["society_id"] = self.society.society_id if self.society else None
        return stats


def run_session83_federation_test(
    num_experts: int = 128,
    num_generations: int = 90,
    num_sequences: int = 9,
    min_trust_evidence: int = 2,
    epsilon: float = 0.2,
    seed: int = 42
):
    """
    Test trust federation between Thor and Legion.

    Scenario:
    1. Thor and Legion run independent trust-first selectors (layer 0)
    2. Thor builds trust through forced exploration (Sessions 77-82)
    3. Thor broadcasts attestations to Legion
    4. Legion imports federated trust with 72% decay (Session 70)
    5. Compare: Legion with federation vs Legion without federation

    Hypothesis: Federation should accelerate trust_driven activation.

    Args:
        num_experts: Number of experts per layer (128 for Q3-Omni)
        num_generations: Number of generations (90 = 9 sequences × 10 epochs)
        num_sequences: Number of distinct sequences
        min_trust_evidence: Minimum observations for trust (Session 78: 2)
        epsilon: Epsilon-greedy exploration rate (Session 77: 0.2)
        seed: Random seed for reproducibility

    Returns:
        Results dictionary
    """
    np.random.seed(seed)

    print("\n" + "="*80)
    print("SESSION 83: TRUST FEDERATION INTEGRATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Experts: {num_experts}")
    print(f"  Generations: {num_generations} ({num_sequences} sequences × {num_generations // num_sequences} epochs)")
    print(f"  Min trust evidence: {min_trust_evidence} (Session 78 optimal)")
    print(f"  Epsilon: {epsilon} (Session 77 optimal)")
    print(f"  Trust decay factor: 0.72 (Session 70)")
    print()

    # Create societies
    thor = Society(
        society_id="thor",
        society_lct="lct://thor-society@testnet/moe",
        secret_key="thor-secret-key-session83",
        platform="Jetson AGX Thor"
    )

    legion = Society(
        society_id="legion",
        society_lct="lct://legion-society@testnet/moe",
        secret_key="legion-secret-key-session83",
        platform="RTX 4090"
    )

    print("Societies:")
    print(f"  1. Thor ({thor.platform})")
    print(f"  2. Legion ({legion.platform})")
    print()

    # Create selectors
    # Thor: Federated (will export attestations)
    thor_selector = FederatedTrustFirstSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet",
        society=thor,
        enable_federation=True
    )

    # Legion WITH federation (imports Thor's attestations)
    legion_fed_selector = FederatedTrustFirstSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet",
        society=legion,
        enable_federation=True
    )

    # Legion WITHOUT federation (baseline, for comparison)
    legion_baseline_selector = TrustFirstMRHSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet"
    )

    # Register societies
    thor_selector.register_society(legion.society_id, legion.secret_key)
    legion_fed_selector.register_society(thor.society_id, thor.secret_key)

    print("Selectors:")
    print(f"  Thor: Federated (exports attestations)")
    print(f"  Legion (federated): Imports Thor's attestations")
    print(f"  Legion (baseline): No federation (comparison)")
    print()

    # Create context classifier (shared for reproducibility)
    context_classifier = ContextClassifier(num_contexts=3)
    thor_selector.context_classifier = context_classifier
    legion_fed_selector.context_classifier = context_classifier
    legion_baseline_selector.context_classifier = context_classifier

    # Fit classifier with diverse sequences
    sequences = [
        "def sort_array(arr):",  # Code (context 0)
        "class BinaryTree:",
        "function process() {",
        "Given the premises, prove:",  # Reasoning (context 1)
        "If A implies B and B implies C,",
        "Solve for x: 3x + 7 = 22",
        "The ancient civilization",  # Text (context 2)
        "Climate change affects",
        "Modern architecture combines"
    ]

    # Generate embeddings (simulated, reproducible)
    embedding_dim = 256
    context_embeddings = []
    for i, seq in enumerate(sequences):
        base_vector = np.random.randn(embedding_dim).astype(np.float32)
        context_group = i // 3  # 0=code, 1=reasoning, 2=text
        base_vector[:10] += context_group * 5.0  # Strong clustering signal
        emb = base_vector / np.linalg.norm(base_vector)
        context_embeddings.append(emb)

    context_classifier.fit(np.array(context_embeddings))

    print("="*80)
    print("SIMULATION: Trust-first with federation")
    print("="*80)
    print()

    # Track results
    thor_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    legion_fed_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    legion_baseline_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    # Simulate generations
    epochs = num_generations // num_sequences

    for gen in range(num_generations):
        seq_idx = gen % num_sequences
        sequence = sequences[seq_idx]
        context = context_classifier.classify(context_embeddings[seq_idx]).context_id

        # Simulate router logits (Q3-Omni style, reproducible)
        np.random.seed(seed + gen)
        router_logits = np.random.randn(num_experts).astype(np.float32)

        # Simulate quality (varies by context, reproducible)
        np.random.seed(seed + gen + 1000)
        quality = np.clip(0.5 + 0.3 * np.random.randn(), 0.0, 1.0)

        # Thor: Select experts and update trust
        thor_result = thor_selector.select_experts(router_logits, context, k=8)
        thor_selector.update_trust_for_expert(
            thor_result.selected_expert_ids[0],
            context,
            quality,
            broadcast=True  # Export attestation
        )

        # Legion (federated): Import Thor's attestations
        for attestation in thor_selector.federation.accepted_attestations:
            legion_fed_selector.import_attestation(
                attestation,
                thor.secret_key
            )

        # Legion (federated): Select experts and update trust
        legion_fed_result = legion_fed_selector.select_experts(router_logits, context, k=8)
        legion_fed_selector.update_trust_for_expert(
            legion_fed_result.selected_expert_ids[0],
            context,
            quality,
            broadcast=False  # Don't re-export (avoid loop)
        )

        # Legion (baseline): Select experts and update trust
        legion_baseline_result = legion_baseline_selector.select_experts(router_logits, context, k=8)
        legion_baseline_selector.update_trust_for_expert(
            legion_baseline_result.selected_expert_ids[0],
            context,
            quality
        )

        # Track results
        for results, selector_result, selector in [
            (thor_results, thor_result, thor_selector),
            (legion_fed_results, legion_fed_result, legion_fed_selector),
            (legion_baseline_results, legion_baseline_result, legion_baseline_selector)
        ]:
            results["mode_history"].append(selector_result.selection_mode)
            results["experts_used"].update(selector_result.selected_expert_ids)

            if selector_result.selection_mode == "trust_driven":
                results["trust_driven_gens"].append(gen)
                if results["first_trust_driven_gen"] is None:
                    results["first_trust_driven_gen"] = gen

    # Compute statistics
    def compute_stats(results, label):
        trust_driven_count = len(results["trust_driven_gens"])
        trust_driven_pct = (trust_driven_count / num_generations) * 100
        experts_used = len(results["experts_used"])
        expert_utilization_pct = (experts_used / num_experts) * 100
        first_activation = results["first_trust_driven_gen"]

        print(f"\n{label}:")
        print(f"  Trust_driven: {trust_driven_count}/{num_generations} ({trust_driven_pct:.1f}%)")
        print(f"  First activation: Gen {first_activation}")
        print(f"  Experts used: {experts_used}/{num_experts} ({expert_utilization_pct:.1f}%)")

        return {
            "trust_driven_count": trust_driven_count,
            "trust_driven_pct": trust_driven_pct,
            "experts_used": experts_used,
            "expert_utilization_pct": expert_utilization_pct,
            "first_activation": first_activation
        }

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    thor_stats = compute_stats(thor_results, "Thor (exports attestations)")
    legion_fed_stats = compute_stats(legion_fed_results, "Legion WITH federation")
    legion_baseline_stats = compute_stats(legion_baseline_results, "Legion WITHOUT federation (baseline)")

    # Federation statistics
    print(f"\nFederation Statistics:")
    thor_fed_stats = thor_selector.get_federation_stats()
    legion_fed_stats_detail = legion_fed_selector.get_federation_stats()

    print(f"\n  Thor:")
    print(f"    Attestations exported: {thor_fed_stats['attestations_exported']}")
    print(f"    Local trust updates: {thor_fed_stats['local_trust_count']}")

    print(f"\n  Legion (federated):")
    print(f"    Attestations imported: {legion_fed_stats_detail['attestations_imported']}")
    print(f"    Attestations rejected: {legion_fed_stats_detail['attestations_rejected']}")
    print(f"    Federated trust applied: {legion_fed_stats_detail['federated_trust_applied']}")
    print(f"    Local trust updates: {legion_fed_stats_detail['local_trust_count']}")

    # Compute federation benefit
    federation_benefit = {
        "trust_driven_improvement": legion_fed_stats["trust_driven_pct"] - legion_baseline_stats["trust_driven_pct"],
        "first_activation_speedup": (legion_baseline_stats["first_activation"] or num_generations) - (legion_fed_stats["first_activation"] or num_generations),
        "expert_diversity_improvement": legion_fed_stats["experts_used"] - legion_baseline_stats["experts_used"]
    }

    print(f"\n" + "="*80)
    print("FEDERATION BENEFIT ANALYSIS")
    print("="*80)
    print(f"\nLegion (federated) vs Legion (baseline):")
    print(f"  Trust_driven improvement: {federation_benefit['trust_driven_improvement']:+.1f}%")
    print(f"  First activation speedup: {federation_benefit['first_activation_speedup']:+d} generations")
    print(f"  Expert diversity improvement: {federation_benefit['expert_diversity_improvement']:+d} experts")

    # Conclusion
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if federation_benefit["trust_driven_improvement"] > 5.0:
        print("\n✅ FEDERATION SUCCESS: Significant trust_driven improvement!")
        print("   Cross-society trust sharing accelerates trust accumulation.")
    elif federation_benefit["trust_driven_improvement"] > 0:
        print("\n✅ FEDERATION BENEFIT: Modest trust_driven improvement.")
        print("   Federation provides value but effect is small.")
    else:
        print("\n⚠️  FEDERATION LIMITED: No trust_driven improvement.")
        print("   Federation may need tuning (decay factor, quorum, etc.)")

    print(f"\n   Key insight: {thor_fed_stats['attestations_exported']} attestations from Thor")
    print(f"   enabled Legion to leverage federated trust ({legion_fed_stats_detail['attestations_imported']} imported).")
    print()

    # Save results
    results = {
        "session": 83,
        "configuration": {
            "num_experts": num_experts,
            "num_generations": num_generations,
            "num_sequences": num_sequences,
            "min_trust_evidence": min_trust_evidence,
            "epsilon": epsilon,
            "trust_decay_factor": 0.72,
            "seed": seed
        },
        "thor": {
            **thor_stats,
            "federation": thor_fed_stats
        },
        "legion_federated": {
            **legion_fed_stats,
            "federation": legion_fed_stats_detail
        },
        "legion_baseline": legion_baseline_stats,
        "federation_benefit": federation_benefit
    }

    output_file = os.path.join(os.path.dirname(__file__), "session83_federation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results


if __name__ == "__main__":
    start_time = time.time()

    results = run_session83_federation_test(
        num_experts=128,
        num_generations=90,  # 9 sequences × 10 epochs
        num_sequences=9,
        min_trust_evidence=2,  # Session 78 optimal
        epsilon=0.2,  # Session 77 optimal
        seed=42
    )

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.1f}s")
    print()
