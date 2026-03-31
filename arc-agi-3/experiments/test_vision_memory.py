#!/usr/bin/env python3
"""
Test Visual Memory in Membot Cartridge

Demonstrates:
- Storing frame snapshots
- Storing action visual outcomes (before/after)
- Computing visual similarity
- Finding similar frames
"""

import sys
sys.path.insert(0, "arc-agi-3/experiments")

import numpy as np
from membot_cartridge import MembotCartridge

print("=" * 70)
print("MEMBOT VISION MEMORY TEST")
print("=" * 70)

# Create test cartridge
cart = MembotCartridge("vision_test")
cart.read()

print("\n1️⃣  Creating test frames...")

# Create test frames (64x64, values 0-15 like ARC grids)
frame_initial = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
frame_rotated_90 = np.rot90(frame_initial)
frame_rotated_180 = np.rot90(frame_initial, 2)
frame_similar = frame_initial.copy()
frame_similar[0:10, 0:10] = 8  # Small change in corner

print("   Created 4 test frames:")
print(f"   - Initial: {frame_initial.shape}, colors: {np.unique(frame_initial).tolist()[:5]}...")
print(f"   - Rotated 90°: shape {frame_rotated_90.shape}")
print(f"   - Rotated 180°: shape {frame_rotated_180.shape}")
print(f"   - Similar (10x10 corner changed): shape {frame_similar.shape}")

# 2. Store frame snapshots
print("\n2️⃣  Storing frame snapshots...")
cart.store_frame_snapshot(
    "level_1_initial",
    frame_initial,
    metadata={"level": 1, "step": 0, "description": "Initial state"}
)
cart.store_frame_snapshot(
    "level_1_rotated_90",
    frame_rotated_90,
    metadata={"level": 1, "step": 10, "description": "After 10 rotations"}
)
cart.store_frame_snapshot(
    "level_1_goal",
    frame_rotated_180,
    metadata={"level": 1, "step": 20, "description": "Goal state"}
)
print("   Stored 3 snapshots: level_1_initial, level_1_rotated_90, level_1_goal")

# 3. Store action visual outcomes
print("\n3️⃣  Storing action visual outcomes...")
cart.store_action_visual_outcome(
    action=6,  # CLICK action
    before_frame=frame_initial,
    after_frame=frame_rotated_90,
    level=1,
    step=5
)
cart.store_action_visual_outcome(
    action=6,
    before_frame=frame_rotated_90,
    after_frame=frame_rotated_180,
    level=1,
    step=15
)
print("   Stored 2 action outcomes (before/after pairs)")

# 4. Compute visual similarity
print("\n4️⃣  Computing visual similarity...")
sim_initial_vs_similar = cart.compute_visual_similarity(frame_initial, frame_similar)
sim_initial_vs_90 = cart.compute_visual_similarity(frame_initial, frame_rotated_90)
sim_initial_vs_180 = cart.compute_visual_similarity(frame_initial, frame_rotated_180)

print(f"   Initial vs Similar (10x10 changed):  {sim_initial_vs_similar:.3f}")
print(f"   Initial vs Rotated 90°:              {sim_initial_vs_90:.3f}")
print(f"   Initial vs Rotated 180°:             {sim_initial_vs_180:.3f}")

# 5. Retrieve a stored frame
print("\n5️⃣  Retrieving stored frame...")
retrieved = cart.get_frame_snapshot("level_1_initial")
if retrieved is not None:
    match = np.array_equal(frame_initial, retrieved)
    print(f"   Retrieved 'level_1_initial': shape {retrieved.shape}, matches original: {match}")
else:
    print("   ERROR: Failed to retrieve frame")

# 6. Find similar snapshots
print("\n6️⃣  Finding similar snapshots...")
similar = cart.find_similar_snapshots(frame_similar, threshold=0.95)
print(f"   Found {len(similar)} snapshots ≥95% similar to 'frame_similar':")
for label, score in similar:
    print(f"      - {label}: {score:.3f}")

# 7. Check visual memory in cartridge
print("\n7️⃣  Visual memory summary...")
snapshots = cart.data.get("visual_memory", {}).get("snapshots", {})
outcomes = cart.data.get("visual_memory", {}).get("action_outcomes", [])
print(f"   Total snapshots stored: {len(snapshots)}")
print(f"   Total action outcomes stored: {len(outcomes)}")

print("\n" + "=" * 70)
print("✅ VISION MEMORY TEST COMPLETE")
print("=" * 70)
print("\nKey capabilities demonstrated:")
print("  ✓ Frame storage (numpy → base64 PNG)")
print("  ✓ Frame retrieval (base64 PNG → numpy)")
print("  ✓ Action outcome tracking (before/after)")
print("  ✓ Visual similarity computation")
print("  ✓ Visual search (find similar frames)")
print("\nNext: Use this for rotation puzzle solving!")
print("  - Store goal states visually")
print("  - Compare current frame to goals")
print("  - Learn which actions move toward goals")
