#!/usr/bin/env python3
"""
CBP Federation Data Pipeline for SAGE Training
Provides cached, witnessed training data from federation activities
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CBPFederationDataset(Dataset):
    """
    CBP's contribution: Real federation data for SAGE training
    Transforms federation messages into training samples
    """

    def __init__(self, federation_inbox: Path, cache_dir: Path):
        self.federation_inbox = federation_inbox
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # CBP's special sauce: Metrics extraction
        self.metrics_extractors = {
            'compliance': self._extract_compliance_metrics,
            'trust': self._extract_trust_metrics,
            'energy': self._extract_energy_metrics,
            'coherence': self._extract_coherence_metrics
        }

        # Load and cache federation data
        self.samples = self._load_federation_data()

    def _load_federation_data(self) -> List[Dict]:
        """Load all federation messages and convert to training samples"""
        samples = []
        cache_file = self.cache_dir / "federation_samples.json"

        # Check cache first (CBP efficiency!)
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if cached['timestamp'] > (datetime.now().timestamp() - 3600):
                    return cached['samples']

        # Process all federation messages
        for msg_file in self.federation_inbox.glob("*.md"):
            try:
                content = msg_file.read_text()
                sample = self._process_message(content, msg_file.stem)
                if sample:
                    samples.append(sample)
            except Exception as e:
                print(f"Error processing {msg_file}: {e}")

        # Cache the results
        cache_data = {
            'timestamp': datetime.now().timestamp(),
            'samples': samples
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        return samples

    def _process_message(self, content: str, msg_id: str) -> Optional[Dict]:
        """Convert federation message to SAGE training sample"""

        # Extract metadata
        lines = content.split('\n')
        metadata = {
            'id': msg_id,
            'society': self._extract_society(msg_id),
            'type': self._classify_message(content),
            'timestamp': datetime.now().isoformat()
        }

        # Extract all metrics (CBP's expertise!)
        metrics = {}
        for name, extractor in self.metrics_extractors.items():
            value = extractor(content)
            if value is not None:
                metrics[name] = value

        # Create training sample
        if metrics:
            return {
                'input': self._create_input_tensor(content, metadata),
                'target': self._create_target_tensor(metrics),
                'metadata': metadata,
                'metrics': metrics,
                'witness': self._generate_witness_hash(content)
            }
        return None

    def _extract_society(self, msg_id: str) -> str:
        """Extract society name from message ID"""
        for society in ['genesis', 'society4', 'society2', 'sprout', 'cbp']:
            if society in msg_id.lower():
                return society
        return 'unknown'

    def _classify_message(self, content: str) -> str:
        """Classify message type based on content"""
        content_lower = content.lower()
        if 'compliance' in content_lower:
            return 'compliance_report'
        elif 'sage' in content_lower:
            return 'sage_update'
        elif 'rfc' in content_lower:
            return 'rfc_proposal'
        elif 'progress' in content_lower:
            return 'progress_report'
        else:
            return 'general'

    def _extract_compliance_metrics(self, content: str) -> Optional[float]:
        """Extract Web4 compliance percentage"""
        import re
        pattern = r'compliance[:\s]+(\d+)%'
        match = re.search(pattern, content.lower())
        if match:
            return float(match.group(1)) / 100.0
        return None

    def _extract_trust_metrics(self, content: str) -> Optional[float]:
        """Extract trust scores"""
        import re
        pattern = r'trust[:\s]+([0-9.]+)'
        match = re.search(pattern, content.lower())
        if match:
            return float(match.group(1))
        return None

    def _extract_energy_metrics(self, content: str) -> Optional[float]:
        """Extract ATP/ADP energy metrics"""
        import re
        pattern = r'atp[:\s]+(\d+)'
        match = re.search(pattern, content.lower())
        if match:
            return float(match.group(1)) / 10000.0  # Normalize
        return None

    def _extract_coherence_metrics(self, content: str) -> Optional[float]:
        """Extract federation coherence"""
        import re
        pattern = r'coherence[:\s]+([0-9.]+)%?'
        match = re.search(pattern, content.lower())
        if match:
            value = float(match.group(1))
            return value / 100.0 if value > 1 else value
        return None

    def _create_input_tensor(self, content: str, metadata: Dict) -> torch.Tensor:
        """Create input tensor for SAGE (simplified for demo)"""
        # In production, this would use proper tokenization
        # For now, simple character-level encoding
        max_len = 512
        chars = list(content[:max_len])
        encoded = [ord(c) % 256 for c in chars]
        padded = encoded + [0] * (max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.float32) / 255.0

    def _create_target_tensor(self, metrics: Dict) -> torch.Tensor:
        """Create target tensor from metrics"""
        # Fixed-size output for SAGE
        target = torch.zeros(4)
        if 'compliance' in metrics:
            target[0] = metrics['compliance']
        if 'trust' in metrics:
            target[1] = metrics['trust']
        if 'energy' in metrics:
            target[2] = metrics['energy']
        if 'coherence' in metrics:
            target[3] = metrics['coherence']
        return target

    def _generate_witness_hash(self, content: str) -> str:
        """Generate witness hash for data integrity"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def get_statistics(self) -> Dict:
        """Get dataset statistics (CBP loves metrics!)"""
        stats = {
            'total_samples': len(self.samples),
            'societies': {},
            'message_types': {},
            'metrics_coverage': {}
        }

        for sample in self.samples:
            # Count by society
            society = sample['metadata']['society']
            stats['societies'][society] = stats['societies'].get(society, 0) + 1

            # Count by type
            msg_type = sample['metadata']['type']
            stats['message_types'][msg_type] = stats['message_types'].get(msg_type, 0) + 1

            # Track metrics coverage
            for metric in sample['metrics']:
                stats['metrics_coverage'][metric] = stats['metrics_coverage'].get(metric, 0) + 1

        return stats


class CBPCacheLayer:
    """
    CBP's caching layer for SAGE inference
    Reduces redundant computations across federation
    """

    def __init__(self, cache_size_mb: int = 100):
        self.cache_size_mb = cache_size_mb
        self.cache = {}
        self.access_counts = {}
        self.last_access = {}

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached result with metrics tracking"""
        if key in self.cache:
            self.access_counts[key] += 1
            self.last_access[key] = datetime.now()
            return self.cache[key]
        return None

    def put(self, key: str, value: torch.Tensor):
        """Store result with cache management"""
        # Simple LRU eviction if needed
        if self._cache_size_mb() > self.cache_size_mb:
            self._evict_lru()

        self.cache[key] = value
        self.access_counts[key] = 1
        self.last_access[key] = datetime.now()

    def _cache_size_mb(self) -> float:
        """Calculate current cache size"""
        size_bytes = sum(
            v.element_size() * v.nelement()
            for v in self.cache.values()
            if isinstance(v, torch.Tensor)
        )
        return size_bytes / (1024 * 1024)

    def _evict_lru(self):
        """Evict least recently used items"""
        if not self.last_access:
            return

        oldest_key = min(self.last_access, key=self.last_access.get)
        del self.cache[oldest_key]
        del self.access_counts[oldest_key]
        del self.last_access[oldest_key]

    def get_metrics(self) -> Dict:
        """Return cache performance metrics"""
        total_accesses = sum(self.access_counts.values())
        return {
            'cache_size_mb': self._cache_size_mb(),
            'num_entries': len(self.cache),
            'total_accesses': total_accesses,
            'hit_rate': len(self.cache) / max(total_accesses, 1),
            'most_accessed': max(self.access_counts, key=self.access_counts.get) if self.access_counts else None
        }


def create_cbp_dataloader(batch_size: int = 32) -> DataLoader:
    """Create DataLoader for SAGE training with CBP data"""
    federation_inbox = Path("/mnt/c/exe/projects/ai-agents/ACT/implementation/ledger/federation_inbox")
    cache_dir = Path("/tmp/cbp_sage_cache")

    dataset = CBPFederationDataset(federation_inbox, cache_dir)

    # Print statistics (CBP loves metrics!)
    stats = dataset.get_statistics()
    print("CBP Federation Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Societies: {stats['societies']}")
    print(f"  Message types: {stats['message_types']}")
    print(f"  Metrics coverage: {stats['metrics_coverage']}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print("🎯 CBP Federation Data Pipeline for SAGE")
    print("=" * 50)

    # Test the pipeline
    dataloader = create_cbp_dataloader(batch_size=16)

    print(f"\n📊 Created DataLoader with {len(dataloader)} batches")

    # Test cache layer
    cache = CBPCacheLayer(cache_size_mb=50)
    print(f"\n💾 Cache Layer initialized with {cache.cache_size_mb}MB limit")

    # Sample batch
    for batch in dataloader:
        print(f"\n🔄 Sample batch:")
        print(f"  Input shape: {batch['input'].shape}")
        print(f"  Target shape: {batch['target'].shape}")
        print(f"  Witnesses: {batch['witness'][:3]}...")
        break

    print("\n✅ CBP Data Pipeline Ready for SAGE Training!")
    print("📡 Federation data → Metrics extraction → Cache optimization → SAGE")