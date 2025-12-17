#!/usr/bin/env python3
"""
Web4 ATP Integration Example

Demonstrates how to use ExpertIdentityBridge and ATPResourceAllocator together
to implement ATP-based cache allocation for SAGE experts.

This example shows:
1. Expert identity registration (expert_id → LCT ID)
2. ATP cost computation based on reputation and cache pressure
3. Cache allocation to highest ATP payers
4. Reward computation based on quality

Use Case: Simulated agent society requesting expert selections with ATP payments

Created: Session 60 (2025-12-16)
Part of: Web4 ↔ SAGE integration
"""

import sys
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web4.expert_identity import ExpertIdentityBridge
from web4.atp_allocator import ATPResourceAllocator


def main():
    """Demonstrate ATP-based expert selection."""

    print("=== Web4 ATP Integration Example ===\n")

    # 1. Initialize components
    print("1. Initializing components...")

    identity_bridge = ExpertIdentityBridge(namespace="sage_legion")
    atp_allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        cache_contention_threshold=0.8,
        max_contention_multiplier=3.0,
        quality_premium_factor=0.5
    )

    print(f"   Identity bridge: {identity_bridge.namespace}")
    print(f"   ATP allocator: base_cost={atp_allocator.base_cost}")
    print()

    # 2. Register experts
    print("2. Registering experts...")

    # Simulate 8 experts with different specializations
    expert_descriptions = {
        0: "Code generation expert",
        1: "Reasoning expert",
        2: "Math expert",
        3: "Creative writing expert",
        4: "Code review expert",
        5: "Debug expert",
        6: "Optimization expert",
        7: "Documentation expert"
    }

    expert_to_lct = identity_bridge.register_batch(
        expert_ids=list(expert_descriptions.keys()),
        descriptions=expert_descriptions
    )

    for expert_id, lct_id in expert_to_lct.items():
        print(f"   Expert {expert_id}: {lct_id}")
    print()

    # 3. Simulate expert reputations
    print("3. Simulating expert reputations...")

    # Expert reputation scores (0-1)
    # In practice, these come from ExpertReputationDB
    expert_reputations = {
        0: 0.92,  # Code generation: high reputation
        1: 0.88,  # Reasoning: high reputation
        2: 0.85,  # Math: high reputation
        3: 0.75,  # Creative writing: good reputation
        4: 0.82,  # Code review: good reputation
        5: 0.70,  # Debug: moderate reputation
        6: 0.78,  # Optimization: good reputation
        7: 0.65   # Documentation: moderate reputation
    }

    for expert_id, reputation in expert_reputations.items():
        print(f"   Expert {expert_id}: reputation={reputation:.2f}")
    print()

    # 4. Compute ATP costs for expert selection
    print("4. Computing ATP costs (cache at 85% utilization)...")

    cache_utilization = 0.85  # Cache is moderately full

    costs = {}
    for expert_id in expert_reputations.keys():
        cost = atp_allocator.compute_cost(
            expert_id=expert_id,
            reputation=expert_reputations[expert_id],
            cache_utilization=cache_utilization
        )
        costs[expert_id] = cost

        # Get cost breakdown
        breakdown = atp_allocator.get_cost_breakdown(expert_id)
        print(f"   Expert {expert_id} (rep={expert_reputations[expert_id]:.2f}): "
              f"{cost} ATP "
              f"(scarcity: {breakdown.scarcity_premium:.2f}, "
              f"quality: {breakdown.quality_premium:.2f})")
    print()

    # 5. Simulate agent requests
    print("5. Simulating agent requests...")

    # Agents request experts with ATP payments
    # Agent strategy: Pay cost + premium based on urgency
    agent_requests = [
        (0, costs[0] + 50),   # Agent 1: Expert 0 (code gen), urgent
        (1, costs[1] + 20),   # Agent 2: Expert 1 (reasoning), normal
        (2, costs[2] + 100),  # Agent 3: Expert 2 (math), very urgent
        (3, costs[3] + 10),   # Agent 4: Expert 3 (creative), low urgency
        (4, costs[4] + 30),   # Agent 5: Expert 4 (code review), normal
        (5, costs[5]),        # Agent 6: Expert 5 (debug), exact cost
        (6, costs[6] + 40),   # Agent 7: Expert 6 (optimization), urgent
        (7, costs[7]),        # Agent 8: Expert 7 (docs), exact cost
    ]

    for expert_id, atp_payment in agent_requests:
        required_cost = costs[expert_id]
        urgency = atp_payment - required_cost
        print(f"   Agent requests expert {expert_id}: "
              f"{atp_payment} ATP (cost={required_cost}, urgency=+{urgency})")
    print()

    # 6. Allocate cache (only 4 experts fit in cache)
    print("6. Allocating cache (capacity: 4 experts)...")

    cache_size = 4  # Simulated cache capacity

    allocated = atp_allocator.allocate_cache(
        requests=agent_requests,
        cache_size=cache_size,
        trust_scores=expert_reputations  # Use reputation as trust
    )

    print(f"   Allocated experts: {allocated}")
    print(f"   Cache utilization: {len(allocated)}/{cache_size} ({len(allocated)/cache_size*100:.0f}%)")

    # Show which agents got their requests fulfilled
    allocated_set = set(allocated)
    for expert_id, atp_payment in agent_requests:
        status = "✓ ALLOCATED" if expert_id in allocated_set else "✗ DENIED"
        print(f"   Expert {expert_id}: {status}")
    print()

    # 7. Simulate generation and compute rewards
    print("7. Simulating generation and computing rewards...")

    # Simulate quality scores for allocated experts
    # In practice, these come from quality_measurement.py
    quality_scores = {
        0: 0.92,  # Code generation: excellent
        1: 0.88,  # Reasoning: excellent
        2: 0.95,  # Math: outstanding
        3: 0.75,  # Creative: good
        4: 0.82,  # Code review: very good
        5: 0.70,  # Debug: acceptable
        6: 0.78,  # Optimization: good
        7: 0.65   # Documentation: acceptable
    }

    total_rewards = 0
    for expert_id in allocated:
        # Find the ATP payment for this expert
        cost_paid = next(payment for eid, payment in agent_requests if eid == expert_id)
        quality = quality_scores[expert_id]

        reward = atp_allocator.compute_reward(
            expert_id=expert_id,
            quality_score=quality,
            cost_paid=cost_paid
        )

        total_rewards += reward
        reward_pct = (reward / cost_paid * 100) if cost_paid > 0 else 0
        print(f"   Expert {expert_id}: quality={quality:.2f} → "
              f"reward={reward} ATP ({reward_pct:.0f}% of cost)")
    print()

    # 8. Show statistics
    print("8. Allocation statistics...")

    stats = atp_allocator.get_statistics()
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Successful allocations: {stats.successful_allocations}")
    print(f"   Total ATP spent: {stats.total_atp_spent}")
    print(f"   Total ATP rewarded: {stats.total_atp_rewarded}")
    print(f"   Net ATP flow: {stats.total_atp_rewarded - stats.total_atp_spent:+d}")
    print(f"   Average cost: {stats.average_cost:.2f}")
    print(f"   Average quality: {stats.average_quality:.2f}")
    print(f"   Cache hit rate: {stats.cache_hit_rate*100:.0f}%")
    print()

    # 9. Economic analysis
    print("9. Economic analysis...")

    atp_efficiency = stats.total_atp_rewarded / max(stats.total_atp_spent, 1)
    print(f"   ATP efficiency: {atp_efficiency:.2%} (rewards/costs)")

    if atp_efficiency > 1.0:
        print(f"   → Agents gained {(atp_efficiency - 1.0)*100:.1f}% surplus (good quality)")
    elif atp_efficiency < 1.0:
        print(f"   → Agents lost {(1.0 - atp_efficiency)*100:.1f}% (poor quality)")
    else:
        print("   → Break-even (costs = rewards)")

    quality_weighted_throughput = stats.average_quality * stats.successful_allocations
    print(f"   Quality-weighted throughput: {quality_weighted_throughput:.2f}")
    print()

    # 10. Show integration benefits
    print("10. Integration benefits...")
    print("   ✓ Expert identity: LCT-based identification")
    print("   ✓ Economic pricing: Scarcity + quality premiums")
    print("   ✓ Resource allocation: ATP-based priority")
    print("   ✓ Quality incentives: Performance-based rewards")
    print("   ✓ Market signals: Price volatility reflects demand")
    print()

    print("=== Integration Example Complete ===")


if __name__ == "__main__":
    main()
