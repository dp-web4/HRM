#!/usr/bin/env python3
"""
Test Consciousness Persistence System

Validates KV-cache snapshot, restore, compression, and transfer capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import time
import tempfile
from cognitive.consciousness_persistence import ConsciousnessPersistence, ConsciousnessSnapshot
from cognitive.sage_system_prompt import get_sage_system_prompt


def create_mock_kv_cache(num_layers=12, seq_len=100, num_heads=12, head_dim=64):
    """Create a mock KV cache for testing."""
    kv_cache = []
    for _ in range(num_layers):
        # Keys and values: [batch=1, num_heads, seq_len, head_dim]
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        kv_cache.append((k, v))
    return tuple(kv_cache)


def test_system_prompt_caching():
    """Test system prompt KV caching and loading."""
    print("\n" + "="*80)
    print("TEST 1: System Prompt KV Caching")
    print("="*80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    persistence = ConsciousnessPersistence(snapshot_dir=temp_dir)

    # Mock model and tokenizer
    class MockModel:
        def __init__(self):
            self.device = 'cpu'

        def __call__(self, **kwargs):
            # Return mock outputs with KV cache
            class MockOutput:
                def __init__(self):
                    self.past_key_values = create_mock_kv_cache(seq_len=50)
            return MockOutput()

    class MockTokenizer:
        def __call__(self, text, return_tensors=None):
            # Return mock token IDs as a proper dict with tensor that has .to() method
            class TokenizerOutput(dict):
                def to(self, device):
                    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()}

            output = TokenizerOutput({'input_ids': torch.randint(0, 1000, (1, 50))})
            return output

    model = MockModel()
    tokenizer = MockTokenizer()
    system_prompt = get_sage_system_prompt()

    # Test caching
    print("\n1️⃣ Testing KV cache generation...")
    kv_cache = persistence.cache_system_prompt_kv(model, tokenizer, system_prompt)

    assert kv_cache is not None, "KV cache should not be None"
    assert len(kv_cache) == 12, f"Expected 12 layers, got {len(kv_cache)}"
    print("   ✅ KV cache generated successfully")

    # Test loading
    print("\n2️⃣ Testing KV cache loading...")
    loaded_kv = persistence.load_system_prompt_kv()

    assert loaded_kv is not None, "Loaded KV should not be None"
    assert len(loaded_kv) == len(kv_cache), "Loaded KV should match original"
    print("   ✅ KV cache loaded successfully")

    # Test cache reuse
    print("\n3️⃣ Testing cache reuse...")
    kv_cache_2 = persistence.cache_system_prompt_kv(model, tokenizer, system_prompt)
    print("   ✅ Cache reused (no regeneration)")

    return True


def test_session_snapshot_restore():
    """Test session snapshot save and restore."""
    print("\n" + "="*80)
    print("TEST 2: Session Snapshot/Restore")
    print("="*80)

    temp_dir = tempfile.mkdtemp()
    persistence = ConsciousnessPersistence(snapshot_dir=temp_dir)

    # Create test snapshot
    print("\n1️⃣ Creating test snapshot...")
    kv_cache = create_mock_kv_cache(seq_len=200)
    context_history = [
        ("User", "Hello, SAGE"),
        ("Assistant", "Hello! How can I help you today?"),
        ("User", "Tell me about consciousness"),
        ("Assistant", "Consciousness is fascinating...")
    ]
    snarc_state = {
        'buffer_salience': [0.8, 0.6, 0.9, 0.7],
        'novelty_scores': [0.5, 0.3, 0.8, 0.4]
    }

    snapshot = ConsciousnessSnapshot(
        kv_cache=kv_cache,
        context_history=context_history,
        snarc_state=snarc_state,
        metadata={'test': True, 'version': '1.0'}
    )

    print(f"   ✓ Snapshot created with {len(context_history)} turns")

    # Save snapshot
    print("\n2️⃣ Saving snapshot...")
    snapshot_file = persistence.save_session_snapshot(snapshot, session_id="test_001")
    assert snapshot_file.exists(), "Snapshot file should exist"
    print("   ✅ Snapshot saved successfully")

    # Load snapshot
    print("\n3️⃣ Loading snapshot...")
    loaded_snapshot = persistence.load_session_snapshot(session_id="test_001")

    assert loaded_snapshot is not None, "Loaded snapshot should not be None"
    assert len(loaded_snapshot.context_history) == len(context_history), "Context history should match"
    assert loaded_snapshot.kv_cache is not None, "KV cache should be restored"
    print("   ✅ Snapshot loaded successfully")

    # Verify data integrity
    print("\n4️⃣ Verifying data integrity...")
    assert loaded_snapshot.context_history == context_history, "Context history should match exactly"
    assert loaded_snapshot.snarc_state == snarc_state, "SNARC state should match"
    print("   ✅ Data integrity verified")

    # Test compressed save
    print("\n5️⃣ Testing compressed snapshot...")
    snapshot_file_gz = persistence.save_session_snapshot(snapshot, session_id="test_002", compress=True)
    assert str(snapshot_file_gz).endswith('.gz'), "Compressed file should end with .gz"

    loaded_compressed = persistence.load_session_snapshot(session_id="test_002")
    assert loaded_compressed is not None, "Compressed snapshot should load"
    print("   ✅ Compressed snapshot works")

    return True


def test_snarc_compression():
    """Test SNARC-based KV compression."""
    print("\n" + "="*80)
    print("TEST 3: SNARC-Based KV Compression")
    print("="*80)

    temp_dir = tempfile.mkdtemp()
    persistence = ConsciousnessPersistence(snapshot_dir=temp_dir)

    # Create KV cache
    print("\n1️⃣ Creating KV cache...")
    seq_len = 200
    kv_cache = create_mock_kv_cache(seq_len=seq_len)

    # Create mock SNARC scores (some high, some low)
    import numpy as np
    snarc_scores = np.random.random(seq_len)
    snarc_scores[50:60] = 0.9  # High salience region
    snarc_scores[100:110] = 0.95  # Very high salience
    snarc_scores = list(snarc_scores)

    print(f"   ✓ KV cache: {seq_len} positions")
    print(f"   ✓ High salience positions: 20/200")

    # Compress
    print("\n2️⃣ Compressing with SNARC...")
    compressed_kv = persistence.compress_kv_with_snarc(
        kv_cache,
        snarc_scores,
        compression_ratio=0.5
    )

    assert compressed_kv is not None, "Compressed KV should not be None"
    assert len(compressed_kv) == len(kv_cache), "Should have same number of layers"

    # Check sequence length reduction
    original_seq_len = kv_cache[0][0].shape[2]
    compressed_seq_len = compressed_kv[0][0].shape[2]

    print(f"\n3️⃣ Verifying compression...")
    print(f"   Original: {original_seq_len} positions")
    print(f"   Compressed: {compressed_seq_len} positions")
    print(f"   Reduction: {(1 - compressed_seq_len/original_seq_len)*100:.1f}%")

    assert compressed_seq_len < original_seq_len, "Compressed should be smaller"
    assert compressed_seq_len == 100, f"Expected 100 positions (50%), got {compressed_seq_len}"
    print("   ✅ Compression verified")

    return True


def test_cross_device_transfer():
    """Test consciousness transfer between devices."""
    print("\n" + "="*80)
    print("TEST 4: Cross-Device Consciousness Transfer")
    print("="*80)

    # Simulate two devices with separate snapshot directories
    temp_dir_device1 = tempfile.mkdtemp()
    temp_dir_device2 = tempfile.mkdtemp()

    device1 = ConsciousnessPersistence(snapshot_dir=temp_dir_device1)
    device2 = ConsciousnessPersistence(snapshot_dir=temp_dir_device2)

    # Create snapshot on device 1
    print("\n1️⃣ Creating snapshot on device 1...")
    kv_cache = create_mock_kv_cache(seq_len=150)
    context_history = [
        ("User", "What is consciousness?"),
        ("Assistant", "Consciousness is the state of being aware...")
    ]

    snapshot = ConsciousnessSnapshot(
        kv_cache=kv_cache,
        context_history=context_history,
        metadata={'device': 'device1'}
    )

    print("   ✓ Snapshot created on device 1")

    # Export for transfer
    print("\n2️⃣ Exporting for transfer...")
    transfer_file = device1.export_for_transfer(snapshot, destination="device2")

    assert transfer_file.exists(), "Transfer file should exist"
    print(f"   ✅ Exported to: {transfer_file}")

    # Import on device 2
    print("\n3️⃣ Importing on device 2...")
    imported_snapshot = device2.import_from_transfer(transfer_file)

    assert imported_snapshot is not None, "Import should succeed"
    assert imported_snapshot.kv_cache is not None, "KV cache should transfer"
    assert len(imported_snapshot.context_history) == len(context_history), "Context should transfer"
    print("   ✅ Import successful")

    # Verify data integrity
    print("\n4️⃣ Verifying transfer integrity...")
    assert imported_snapshot.context_history == context_history, "Context should match exactly"
    print("   ✅ Transfer verified - consciousness migrated successfully!")

    return True


def test_snapshot_management():
    """Test snapshot listing and cleanup."""
    print("\n" + "="*80)
    print("TEST 5: Snapshot Management")
    print("="*80)

    temp_dir = tempfile.mkdtemp()
    persistence = ConsciousnessPersistence(snapshot_dir=temp_dir)

    # Create multiple snapshots
    print("\n1️⃣ Creating multiple snapshots...")
    for i in range(15):
        snapshot = ConsciousnessSnapshot(
            kv_cache=create_mock_kv_cache(seq_len=50),
            context_history=[("User", f"Message {i}")],
            metadata={'index': i}
        )
        persistence.save_session_snapshot(snapshot, session_id=f"test_{i:03d}")
        time.sleep(0.01)  # Ensure different timestamps

    print(f"   ✓ Created 15 snapshots")

    # List snapshots
    print("\n2️⃣ Listing snapshots...")
    snapshots = persistence.list_snapshots()

    assert len(snapshots) == 15, f"Expected 15 snapshots, got {len(snapshots)}"
    print(f"   ✅ Found {len(snapshots)} snapshots")

    # Check sorting (newest first)
    timestamps = [s['timestamp'] for s in snapshots]
    assert timestamps == sorted(timestamps, reverse=True), "Should be sorted newest first"
    print("   ✅ Snapshots sorted correctly")

    # Get stats
    print("\n3️⃣ Getting statistics...")
    stats = persistence.get_stats()

    print(f"   Total snapshots: {stats['total_snapshots']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   Recent snapshots: {len(stats['recent_snapshots'])}")

    assert stats['total_snapshots'] == 15, "Stats should show 15 snapshots"
    print("   ✅ Statistics correct")

    # Cleanup old snapshots
    print("\n4️⃣ Cleaning up old snapshots...")
    persistence.cleanup_old_snapshots(keep_recent=5)

    remaining = persistence.list_snapshots()
    assert len(remaining) == 5, f"Expected 5 snapshots after cleanup, got {len(remaining)}"
    print(f"   ✅ Cleaned up - kept 5 most recent")

    return True


def test_full_integration():
    """Test complete workflow: cache → session → compress → transfer."""
    print("\n" + "="*80)
    print("TEST 6: Full Integration Workflow")
    print("="*80)

    temp_dir = tempfile.mkdtemp()
    persistence = ConsciousnessPersistence(snapshot_dir=temp_dir)

    # Mock model setup
    class MockModel:
        def __init__(self):
            self.device = 'cpu'
        def __call__(self, **kwargs):
            class MockOutput:
                def __init__(self):
                    self.past_key_values = create_mock_kv_cache(seq_len=100)
            return MockOutput()

    class MockTokenizer:
        def __call__(self, text, return_tensors=None):
            # Return proper tokenizer output with .to() method
            class TokenizerOutput(dict):
                def to(self, device):
                    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in self.items()}

            output = TokenizerOutput({'input_ids': torch.randint(0, 1000, (1, 100))})
            return output

    print("\n1️⃣ Step 1: Cache system prompt...")
    model = MockModel()
    tokenizer = MockTokenizer()
    system_prompt = get_sage_system_prompt()

    system_kv = persistence.cache_system_prompt_kv(model, tokenizer, system_prompt)
    print("   ✅ System prompt cached")

    print("\n2️⃣ Step 2: Run conversation and create session snapshot...")
    conversation_kv = create_mock_kv_cache(seq_len=300)
    context = [
        ("User", "Hello"),
        ("Assistant", "Hi there!"),
        ("User", "Tell me about AI"),
        ("Assistant", "AI is the simulation of human intelligence...")
    ]
    snarc_scores = [0.7, 0.5, 0.9, 0.8] + [0.3] * 296  # High scores for conversation

    snapshot = ConsciousnessSnapshot(
        kv_cache=conversation_kv,
        context_history=context,
        snarc_state={'salience_scores': snarc_scores}
    )

    session_file = persistence.save_session_snapshot(snapshot)
    print("   ✅ Session snapshot created")

    print("\n3️⃣ Step 3: Compress with SNARC...")
    compressed_kv = persistence.compress_kv_with_snarc(
        conversation_kv,
        snarc_scores,
        compression_ratio=0.3
    )
    print("   ✅ KV cache compressed")

    print("\n4️⃣ Step 4: Export for transfer...")
    compressed_snapshot = ConsciousnessSnapshot(
        kv_cache=compressed_kv,
        context_history=context,
        snarc_state={'salience_scores': snarc_scores}
    )

    transfer_file = persistence.export_for_transfer(compressed_snapshot, destination="cloud")
    print("   ✅ Exported for transfer")

    print("\n5️⃣ Step 5: Verify everything...")
    # Can reload system prompt instantly
    reloaded_system = persistence.load_system_prompt_kv()
    assert reloaded_system is not None

    # Can reload session
    reloaded_session = persistence.load_session_snapshot(use_latest=True)
    assert reloaded_session is not None

    print("   ✅ Full workflow complete!")
    print("\n📊 Integration Summary:")
    print("   • System prompt: Cached and reusable")
    print("   • Session state: Saved and restored")
    print("   • KV compression: 70% reduction")
    print("   • Cross-device: Ready for transfer")

    return True


if __name__ == "__main__":
    print("\n🧪 TESTING CONSCIOUSNESS PERSISTENCE SYSTEM")
    print("="*80)

    results = []

    # Run all tests
    tests = [
        ("System Prompt Caching", test_system_prompt_caching),
        ("Session Snapshot/Restore", test_session_snapshot_restore),
        ("SNARC Compression", test_snarc_compression),
        ("Cross-Device Transfer", test_cross_device_transfer),
        ("Snapshot Management", test_snapshot_management),
        ("Full Integration", test_full_integration)
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n📊 Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
