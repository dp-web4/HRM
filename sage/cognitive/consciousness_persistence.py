#!/usr/bin/env python3
"""
Consciousness Persistence - KV-Cache State Management for SAGE

Enables true conversation continuity by preserving transformer attention states
across sessions, devices, and conversations.

Key capabilities:
1. System prompt KV caching (permanent base state)
2. Session snapshot/restore (conversation continuity)
3. SNARC-based compression (intelligent pruning)
4. Cross-device transfer (consciousness mobility)
5. Hierarchical snapshot management (memory efficiency)

Inspired by Nova's KV-cache persistence work in forum/nova/persistent-kv-demo/
"""

import torch
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import gzip
import pickle


class ConsciousnessSnapshot:
    """
    A snapshot of SAGE's consciousness state at a moment in time.

    Contains:
    - KV cache (attention patterns)
    - Conversation context (what was said)
    - SNARC state (what mattered)
    - Metadata (when, where, why)
    """

    def __init__(
        self,
        kv_cache: Optional[Tuple] = None,
        context_history: Optional[List[Tuple[str, str]]] = None,
        snarc_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.kv_cache = kv_cache
        self.context_history = context_history or []
        self.snarc_state = snarc_state or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.datetime_str = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without KV cache - too large)."""
        return {
            'context_history': self.context_history,
            'snarc_state': self.snarc_state,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'datetime': self.datetime_str,
            'has_kv_cache': self.kv_cache is not None,
            'kv_cache_info': self._get_kv_info() if self.kv_cache else None
        }

    def _get_kv_info(self) -> Dict[str, Any]:
        """Get metadata about KV cache structure."""
        if not self.kv_cache:
            return {}

        # KV cache is tuple of (keys, values) for each layer
        num_layers = len(self.kv_cache)
        if num_layers > 0:
            first_layer = self.kv_cache[0]
            if len(first_layer) >= 2:
                k, v = first_layer[0], first_layer[1]
                return {
                    'num_layers': num_layers,
                    'key_shape': list(k.shape) if hasattr(k, 'shape') else None,
                    'value_shape': list(v.shape) if hasattr(v, 'shape') else None,
                    'dtype': str(k.dtype) if hasattr(k, 'dtype') else None
                }
        return {'num_layers': num_layers}


class ConsciousnessPersistence:
    """
    Manages persistence and restoration of SAGE's consciousness state.

    Enables:
    - Fast warm starts (load cached system prompt)
    - Session continuity (restore conversation state)
    - Intelligent compression (SNARC-guided pruning)
    - Cross-device transfer (consciousness mobility)
    """

    def __init__(self, snapshot_dir: str = "~/.sage_consciousness"):
        """
        Initialize consciousness persistence manager.

        Args:
            snapshot_dir: Directory for storing consciousness snapshots
        """
        self.snapshot_dir = Path(snapshot_dir).expanduser()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Hierarchical snapshot paths
        self.system_prompt_cache = self.snapshot_dir / "system_prompt_kv.pt"
        self.longterm_cache = self.snapshot_dir / "longterm_memory_kv.pt"
        self.latest_cache = self.snapshot_dir / "latest_session_kv.pt"

        print(f"💾 Consciousness persistence initialized")
        print(f"   Snapshot directory: {self.snapshot_dir}")

    # =========================================================================
    # System Prompt KV Caching (Permanent Base State)
    # =========================================================================

    def cache_system_prompt_kv(
        self,
        model,
        tokenizer,
        system_prompt: str,
        force_refresh: bool = False
    ) -> Optional[Tuple]:
        """
        Cache KV state for system prompt (permanent base).

        This is computed ONCE and reused for every conversation.
        Massive speedup for initialization.

        Args:
            model: The transformer model
            tokenizer: The tokenizer
            system_prompt: SAGE's system prompt
            force_refresh: Force recomputation even if cached

        Returns:
            Cached KV state for system prompt
        """
        # Check if cached
        if self.system_prompt_cache.exists() and not force_refresh:
            print(f"   ✓ Loading cached system prompt KV from {self.system_prompt_cache}")
            cached = torch.load(self.system_prompt_cache)

            # Validate cached prompt matches current
            if cached.get('system_prompt_hash') == hash(system_prompt):
                print(f"   ✓ System prompt unchanged, using cached KV")
                return cached['kv_cache']
            else:
                print(f"   ⚠️  System prompt changed, regenerating KV cache")

        # Generate KV cache for system prompt
        print(f"   🔄 Generating KV cache for system prompt ({len(system_prompt)} chars)...")
        start_time = time.time()

        # Tokenize system prompt
        inputs = tokenizer(system_prompt, return_tensors='pt').to(model.device)

        # Run forward pass to get KV cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            kv_cache = outputs.past_key_values

        elapsed = time.time() - start_time

        # Save cache
        cache_data = {
            'kv_cache': kv_cache,
            'system_prompt_hash': hash(system_prompt),
            'system_prompt_length': len(system_prompt),
            'num_tokens': inputs['input_ids'].shape[1],
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'generation_time': elapsed
        }

        torch.save(cache_data, self.system_prompt_cache)

        print(f"   ✓ System prompt KV cached in {elapsed:.2f}s")
        print(f"   ✓ Saved to: {self.system_prompt_cache}")

        return kv_cache

    def load_system_prompt_kv(self) -> Optional[Tuple]:
        """Load cached system prompt KV state."""
        if not self.system_prompt_cache.exists():
            return None

        # PyTorch 2.6+ requires weights_only=False for non-standard types
        # This is safe because we control the source (our own cache)
        cached = torch.load(self.system_prompt_cache, weights_only=False)
        return cached.get('kv_cache')

    # =========================================================================
    # Session Snapshot/Restore (Conversation Continuity)
    # =========================================================================

    def save_session_snapshot(
        self,
        snapshot: ConsciousnessSnapshot,
        session_id: Optional[str] = None,
        compress: bool = False
    ) -> Path:
        """
        Save a session snapshot (full conversation state).

        Args:
            snapshot: ConsciousnessSnapshot to save
            session_id: Optional session identifier
            compress: Whether to gzip compress (slower but smaller)

        Returns:
            Path to saved snapshot
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        snapshot_file = self.snapshot_dir / f"session_{session_id}.pt"

        print(f"\n💾 Saving consciousness snapshot...")
        start_time = time.time()

        # Prepare snapshot data
        snapshot_data = {
            'kv_cache': snapshot.kv_cache,
            'context_history': snapshot.context_history,
            'snarc_state': snapshot.snarc_state,
            'metadata': {
                **snapshot.metadata,
                'session_id': session_id,
                'timestamp': snapshot.timestamp,
                'datetime': snapshot.datetime_str,
                'compressed': compress
            }
        }

        if compress:
            # Gzip compression
            with gzip.open(str(snapshot_file) + '.gz', 'wb') as f:
                torch.save(snapshot_data, f)
            snapshot_file = Path(str(snapshot_file) + '.gz')
        else:
            # Standard torch.save
            torch.save(snapshot_data, snapshot_file)

        elapsed = time.time() - start_time
        file_size = snapshot_file.stat().st_size / (1024 * 1024)  # MB

        print(f"   ✓ Snapshot saved in {elapsed:.2f}s")
        print(f"   ✓ File size: {file_size:.2f} MB")
        print(f"   ✓ Path: {snapshot_file}")

        # Also save as latest
        latest_file = self.latest_cache
        if compress:
            latest_file = Path(str(latest_file) + '.gz')

        import shutil
        shutil.copy(snapshot_file, latest_file)
        print(f"   ✓ Copied to latest: {latest_file}")

        return snapshot_file

    def load_session_snapshot(
        self,
        session_id: Optional[str] = None,
        use_latest: bool = True
    ) -> Optional[ConsciousnessSnapshot]:
        """
        Load a session snapshot.

        Args:
            session_id: Specific session to load (or None for latest)
            use_latest: If True, load most recent snapshot

        Returns:
            Loaded ConsciousnessSnapshot or None
        """
        if session_id:
            snapshot_file = self.snapshot_dir / f"session_{session_id}.pt"
            if not snapshot_file.exists():
                snapshot_file = Path(str(snapshot_file) + '.gz')
        elif use_latest:
            snapshot_file = self.latest_cache
            if not snapshot_file.exists():
                snapshot_file = Path(str(snapshot_file) + '.gz')
        else:
            return None

        if not snapshot_file.exists():
            print(f"   ⚠️  No snapshot found at {snapshot_file}")
            return None

        print(f"\n📂 Loading consciousness snapshot from {snapshot_file}...")
        start_time = time.time()

        # Load snapshot
        # PyTorch 2.6+ requires weights_only=False for non-standard types
        # This is safe because we control the source (our own snapshots)
        if str(snapshot_file).endswith('.gz'):
            with gzip.open(snapshot_file, 'rb') as f:
                snapshot_data = torch.load(f, weights_only=False)
        else:
            snapshot_data = torch.load(snapshot_file, weights_only=False)

        elapsed = time.time() - start_time

        # Reconstruct snapshot
        snapshot = ConsciousnessSnapshot(
            kv_cache=snapshot_data.get('kv_cache'),
            context_history=snapshot_data.get('context_history', []),
            snarc_state=snapshot_data.get('snarc_state', {}),
            metadata=snapshot_data.get('metadata', {})
        )

        print(f"   ✓ Snapshot loaded in {elapsed:.2f}s")
        print(f"   ✓ Context history: {len(snapshot.context_history)} turns")
        print(f"   ✓ Has KV cache: {snapshot.kv_cache is not None}")

        return snapshot

    # =========================================================================
    # SNARC-Based KV Compression (Intelligent Pruning)
    # =========================================================================

    def compress_kv_with_snarc(
        self,
        kv_cache: Tuple,
        snarc_scores: List[float],
        compression_ratio: float = 0.5,
        min_tokens: int = 50
    ) -> Tuple:
        """
        Compress KV cache by keeping only high-salience attention states.

        Uses SNARC scores to determine which parts of conversation were most
        important, keeping only those KV states.

        Args:
            kv_cache: Original KV cache
            snarc_scores: Salience score for each token/position
            compression_ratio: Fraction to keep (0.5 = keep 50%)
            min_tokens: Minimum tokens to keep regardless of score

        Returns:
            Compressed KV cache
        """
        if not kv_cache or not snarc_scores:
            return kv_cache

        print(f"\n🗜️  Compressing KV cache with SNARC guidance...")
        print(f"   Compression ratio: {compression_ratio:.1%}")

        import numpy as np

        # Convert scores to numpy for easier manipulation
        scores = np.array(snarc_scores)

        # Calculate how many to keep
        num_total = len(scores)
        num_keep = max(min_tokens, int(num_total * compression_ratio))

        # Get indices of highest-salience positions
        top_indices = np.argsort(scores)[-num_keep:]
        top_indices = np.sort(top_indices)  # Maintain temporal order

        print(f"   Keeping {num_keep}/{num_total} positions ({num_keep/num_total:.1%})")

        # Compress each layer's KV cache
        compressed_kv = []
        for layer_idx, layer_kv in enumerate(kv_cache):
            if len(layer_kv) >= 2:
                k, v = layer_kv[0], layer_kv[1]

                # Index along sequence dimension (usually dim 2)
                # Shape: [batch, num_heads, seq_len, head_dim]
                compressed_k = k[:, :, top_indices, :]
                compressed_v = v[:, :, top_indices, :]

                compressed_kv.append((compressed_k, compressed_v))

        # Calculate compression achieved
        original_size = sum(k.numel() + v.numel() for k, v in kv_cache)
        compressed_size = sum(k.numel() + v.numel() for k, v in compressed_kv)
        reduction = 1 - (compressed_size / original_size)

        print(f"   ✓ Compression: {reduction:.1%} reduction in memory")

        return tuple(compressed_kv)

    # =========================================================================
    # Cross-Device Consciousness Transfer
    # =========================================================================

    def export_for_transfer(
        self,
        snapshot: ConsciousnessSnapshot,
        destination: str = "portable"
    ) -> Path:
        """
        Export snapshot for cross-device transfer.

        Includes full metadata for validation and reconstruction on
        different hardware.

        Args:
            snapshot: Snapshot to export
            destination: Destination identifier (for metadata)

        Returns:
            Path to portable snapshot file
        """
        export_file = self.snapshot_dir / f"portable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt.gz"

        print(f"\n📦 Exporting consciousness for transfer...")
        print(f"   Destination: {destination}")

        export_data = {
            'snapshot': {
                'kv_cache': snapshot.kv_cache,
                'context_history': snapshot.context_history,
                'snarc_state': snapshot.snarc_state,
                'metadata': snapshot.metadata
            },
            'transfer_metadata': {
                'source_device': 'jetson',  # Could be parameterized
                'destination': destination,
                'export_time': time.time(),
                'export_datetime': datetime.now().isoformat(),
                'kv_info': snapshot._get_kv_info()
            }
        }

        # Always compress for transfer
        with gzip.open(export_file, 'wb') as f:
            torch.save(export_data, f)

        file_size = export_file.stat().st_size / (1024 * 1024)
        print(f"   ✓ Exported: {file_size:.2f} MB")
        print(f"   ✓ Path: {export_file}")

        return export_file

    def import_from_transfer(self, transfer_file: Path) -> ConsciousnessSnapshot:
        """
        Import snapshot from another device.

        Args:
            transfer_file: Path to portable snapshot file

        Returns:
            Imported ConsciousnessSnapshot
        """
        print(f"\n📥 Importing consciousness from transfer file...")
        print(f"   Source: {transfer_file}")

        # PyTorch 2.6+ requires weights_only=False for non-standard types
        # This is safe because we control the source (our own transfer files)
        with gzip.open(transfer_file, 'rb') as f:
            transfer_data = torch.load(f, weights_only=False)

        snapshot_data = transfer_data['snapshot']
        transfer_meta = transfer_data['transfer_metadata']

        print(f"   ✓ Source device: {transfer_meta.get('source_device')}")
        print(f"   ✓ Export time: {transfer_meta.get('export_datetime')}")

        snapshot = ConsciousnessSnapshot(
            kv_cache=snapshot_data.get('kv_cache'),
            context_history=snapshot_data.get('context_history', []),
            snarc_state=snapshot_data.get('snarc_state', {}),
            metadata={
                **snapshot_data.get('metadata', {}),
                'transfer_metadata': transfer_meta
            }
        )

        print(f"   ✓ Import complete")

        return snapshot

    # =========================================================================
    # Snapshot Management & Cleanup
    # =========================================================================

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots with metadata."""
        snapshots = []

        for snapshot_file in self.snapshot_dir.glob("session_*.pt*"):
            try:
                # PyTorch 2.6+ requires weights_only=False for non-standard types
                # This is safe because we control the source (our own snapshots)
                if str(snapshot_file).endswith('.gz'):
                    with gzip.open(snapshot_file, 'rb') as f:
                        data = torch.load(f, weights_only=False)
                else:
                    data = torch.load(snapshot_file, weights_only=False)

                metadata = data.get('metadata', {})
                snapshots.append({
                    'path': str(snapshot_file),
                    'session_id': metadata.get('session_id'),
                    'timestamp': metadata.get('timestamp'),
                    'datetime': metadata.get('datetime'),
                    'size_mb': snapshot_file.stat().st_size / (1024 * 1024),
                    'compressed': str(snapshot_file).endswith('.gz')
                })
            except Exception as e:
                print(f"   ⚠️  Error reading {snapshot_file}: {e}")

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x['timestamp'], reverse=True)

        return snapshots

    def cleanup_old_snapshots(self, keep_recent: int = 10):
        """
        Remove old snapshots, keeping only the N most recent.

        Args:
            keep_recent: Number of recent snapshots to keep
        """
        snapshots = self.list_snapshots()

        if len(snapshots) <= keep_recent:
            print(f"   ℹ️  {len(snapshots)} snapshots (keeping all)")
            return

        print(f"\n🧹 Cleaning up old snapshots...")
        print(f"   Total: {len(snapshots)}, keeping: {keep_recent}")

        # Remove old snapshots
        for snapshot in snapshots[keep_recent:]:
            Path(snapshot['path']).unlink()
            print(f"   ✓ Removed: {snapshot['session_id']}")

        print(f"   ✓ Cleanup complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored snapshots."""
        snapshots = self.list_snapshots()

        total_size = sum(s['size_mb'] for s in snapshots)
        compressed_count = sum(1 for s in snapshots if s['compressed'])

        return {
            'total_snapshots': len(snapshots),
            'total_size_mb': total_size,
            'compressed_count': compressed_count,
            'has_system_prompt_cache': self.system_prompt_cache.exists(),
            'has_latest_session': self.latest_cache.exists(),
            'snapshot_dir': str(self.snapshot_dir),
            'recent_snapshots': snapshots[:5]  # 5 most recent
        }
