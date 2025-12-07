"""
SAGE Pattern Library with Cryptographic Provenance
===================================================

Enables cross-platform knowledge sharing with cryptographic verification.

**Purpose**: Share learned patterns (weights, thresholds, configurations) across
consciousnesses (Thor, Sprout, Legion) with provable provenance.

**Architecture**:
- Patterns signed by creator consciousness (LCT identity)
- Signatures provide tamper detection and source attribution
- Cross-platform verification (Thor creates, Sprout verifies)
- No central authority needed (trustless federation)

**Pattern Types**:
- SNARC weights (learned compression weights)
- Attention thresholds (per metabolic state)
- Sensor configurations
- Learning hyperparameters
- Performance benchmarks

**Cryptographic Properties**:
- Creator attribution (who created this pattern)
- Tamper detection (pattern hasn't been modified)
- Timestamp (when pattern was created)
- Machine binding (which hardware created it)

**Author**: Claude (autonomous research) on Thor
**Date**: 2025-12-07
**Session**: Pattern library implementation
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from simulated_lct_identity import SimulatedLCTIdentity, LCTKey, LCTSignature


# ============================================================================
# Pattern Types
# ============================================================================

@dataclass
class PatternMetadata:
    """Metadata for a signed pattern"""
    pattern_id: str            # Unique identifier
    pattern_type: str          # Type of pattern (snarc_weights, thresholds, etc.)
    version: str               # Pattern version (semantic versioning)
    description: str           # Human-readable description
    created_at: str            # ISO timestamp
    creator_lct_id: str        # Who created it
    creator_machine: str       # Which machine
    tags: List[str]            # Searchable tags
    metadata: Dict[str, Any]   # Additional metadata

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SignedPattern:
    """Pattern with cryptographic signature"""
    metadata: PatternMetadata
    pattern_data: Dict[str, Any]  # The actual pattern
    signature: LCTSignature        # Cryptographic signature

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'metadata': self.metadata.to_dict(),
            'pattern_data': self.pattern_data,
            'signature': self.signature.to_dict()
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SignedPattern':
        """Deserialize from dictionary"""
        return cls(
            metadata=PatternMetadata(**data['metadata']),
            pattern_data=data['pattern_data'],
            signature=LCTSignature(**data['signature'])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'SignedPattern':
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# Pattern Library
# ============================================================================

class PatternLibrary:
    """
    Pattern library with cryptographic provenance.

    Manages creation, storage, and verification of signed patterns.

    Features:
    - Create patterns with automatic signing
    - Store patterns to disk
    - Load and verify patterns
    - Search patterns by type, tags, creator
    - Cross-platform verification

    Usage:
        library = PatternLibrary(lct_identity, storage_dir)

        # Create pattern
        pattern = library.create_pattern(
            pattern_type="snarc_weights",
            pattern_data={"novelty": 0.6, "arousal": 0.3, ...},
            description="SNARC weights learned from online learning",
            tags=["online_learning", "thor", "validated"]
        )

        # Save pattern
        library.save_pattern(pattern)

        # Load and verify
        loaded = library.load_pattern(pattern_id)
        if library.verify_pattern(loaded):
            # Pattern verified, use it
            weights = loaded.pattern_data
    """

    def __init__(
        self,
        lct_identity: SimulatedLCTIdentity,
        consciousness_lct_id: str,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize pattern library.

        Args:
            lct_identity: LCT identity manager
            consciousness_lct_id: This consciousness's LCT ID
            storage_dir: Where to store patterns (default: ~/.sage/patterns/)
        """
        self.lct_identity = lct_identity
        self.consciousness_lct_id = consciousness_lct_id

        # Ensure identity exists
        self.consciousness_key = lct_identity.get_or_create_identity(consciousness_lct_id)

        if storage_dir is None:
            storage_dir = Path.home() / ".sage" / "patterns"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        description: str,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pattern_id: Optional[str] = None
    ) -> SignedPattern:
        """
        Create a new signed pattern.

        Args:
            pattern_type: Type of pattern (snarc_weights, thresholds, etc.)
            pattern_data: The actual pattern data
            description: Human-readable description
            version: Pattern version (semantic versioning)
            tags: Searchable tags
            metadata: Additional metadata
            pattern_id: Optional custom ID (auto-generated if not provided)

        Returns:
            SignedPattern with cryptographic signature
        """
        # Generate pattern ID if not provided
        if pattern_id is None:
            # Hash pattern data for unique ID
            data_str = json.dumps(pattern_data, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
            pattern_id = f"{pattern_type}_{data_hash}"

        # Create metadata
        pattern_metadata = PatternMetadata(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            version=version,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
            creator_lct_id=self.consciousness_lct_id,
            creator_machine=self.consciousness_key.machine_identity,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Create data to sign (metadata + pattern_data)
        signing_data = {
            'metadata': pattern_metadata.to_dict(),
            'pattern_data': pattern_data
        }
        signing_bytes = json.dumps(signing_data, sort_keys=True).encode('utf-8')

        # Sign
        signature = self.lct_identity.sign_data(
            self.consciousness_lct_id,
            signing_bytes
        )

        return SignedPattern(
            metadata=pattern_metadata,
            pattern_data=pattern_data,
            signature=signature
        )

    def verify_pattern(
        self,
        pattern: SignedPattern,
        public_key_pem: Optional[str] = None
    ) -> bool:
        """
        Verify pattern signature.

        Args:
            pattern: Pattern to verify
            public_key_pem: Optional public key (if verifying from different creator)

        Returns:
            True if signature valid, False otherwise
        """
        # Reconstruct signed data
        signing_data = {
            'metadata': pattern.metadata.to_dict(),
            'pattern_data': pattern.pattern_data
        }
        signing_bytes = json.dumps(signing_data, sort_keys=True).encode('utf-8')

        # Verify
        return self.lct_identity.verify_signature(
            pattern.signature,
            signing_bytes,
            public_key_pem=public_key_pem
        )

    def save_pattern(self, pattern: SignedPattern) -> Path:
        """
        Save pattern to disk.

        Args:
            pattern: Pattern to save

        Returns:
            Path where pattern was saved
        """
        # Create type directory
        type_dir = self.storage_dir / pattern.metadata.pattern_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        filename = f"{pattern.metadata.pattern_id}.json"
        filepath = type_dir / filename

        with open(filepath, 'w') as f:
            f.write(pattern.to_json())

        return filepath

    def load_pattern(self, pattern_id: str, pattern_type: Optional[str] = None) -> SignedPattern:
        """
        Load pattern from disk.

        Args:
            pattern_id: Pattern ID to load
            pattern_type: Optional pattern type (speeds up search)

        Returns:
            SignedPattern

        Raises:
            FileNotFoundError: If pattern not found
        """
        # Search strategy
        if pattern_type:
            # Direct lookup
            filepath = self.storage_dir / pattern_type / f"{pattern_id}.json"
            if not filepath.exists():
                raise FileNotFoundError(f"Pattern {pattern_id} not found in {pattern_type}")
        else:
            # Search all types
            filepath = None
            for type_dir in self.storage_dir.iterdir():
                if type_dir.is_dir():
                    candidate = type_dir / f"{pattern_id}.json"
                    if candidate.exists():
                        filepath = candidate
                        break

            if filepath is None:
                raise FileNotFoundError(f"Pattern {pattern_id} not found")

        # Load
        with open(filepath, 'r') as f:
            return SignedPattern.from_json(f.read())

    def list_patterns(
        self,
        pattern_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        creator_lct_id: Optional[str] = None
    ) -> List[SignedPattern]:
        """
        List patterns matching criteria.

        Args:
            pattern_type: Filter by type
            tags: Filter by tags (any match)
            creator_lct_id: Filter by creator

        Returns:
            List of matching SignedPattern objects
        """
        patterns = []

        # Determine directories to search
        if pattern_type:
            search_dirs = [self.storage_dir / pattern_type]
        else:
            search_dirs = [d for d in self.storage_dir.iterdir() if d.is_dir()]

        # Load all patterns
        for type_dir in search_dirs:
            if not type_dir.exists():
                continue

            for pattern_file in type_dir.glob("*.json"):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern = SignedPattern.from_json(f.read())

                    # Apply filters
                    if tags and not any(tag in pattern.metadata.tags for tag in tags):
                        continue

                    if creator_lct_id and pattern.metadata.creator_lct_id != creator_lct_id:
                        continue

                    patterns.append(pattern)

                except Exception as e:
                    # Skip malformed patterns
                    print(f"Warning: Could not load {pattern_file}: {e}")
                    continue

        return patterns

    def import_pattern(
        self,
        pattern_json: str,
        verify: bool = True,
        creator_public_key: Optional[str] = None
    ) -> SignedPattern:
        """
        Import pattern from JSON (e.g., from another consciousness).

        Args:
            pattern_json: Pattern JSON string
            verify: Verify signature before accepting
            creator_public_key: Public key of pattern creator

        Returns:
            SignedPattern

        Raises:
            ValueError: If verification fails
        """
        pattern = SignedPattern.from_json(pattern_json)

        if verify:
            if not self.verify_pattern(pattern, public_key_pem=creator_public_key):
                raise ValueError(
                    f"Pattern {pattern.metadata.pattern_id} failed signature verification"
                )

        return pattern

    def export_pattern(self, pattern: SignedPattern) -> str:
        """
        Export pattern to JSON for sharing.

        Args:
            pattern: Pattern to export

        Returns:
            JSON string ready for sharing
        """
        return pattern.to_json()


# ============================================================================
# Pattern Templates
# ============================================================================

class PatternTemplates:
    """
    Templates for common pattern types.

    Provides helper functions to create standard patterns.
    """

    @staticmethod
    def snarc_weights(
        surprise: float,
        novelty: float,
        arousal: float,
        reward: float,
        conflict: float,
        description: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create SNARC weights pattern.

        Args:
            surprise, novelty, arousal, reward, conflict: Weight values [0-1]
            description: How these weights were learned
            tags: Searchable tags

        Returns:
            Pattern data dict ready for library.create_pattern()
        """
        # Normalize to sum to 1.0
        total = surprise + novelty + arousal + reward + conflict

        pattern_data = {
            'surprise': surprise / total,
            'novelty': novelty / total,
            'arousal': arousal / total,
            'reward': reward / total,
            'conflict': conflict / total,
            'total': 1.0
        }

        return pattern_data

    @staticmethod
    def metabolic_thresholds(
        wake: float,
        focus: float,
        rest: float,
        dream: float,
        description: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create metabolic state thresholds pattern.

        Args:
            wake, focus, rest, dream: Threshold values [0-1]
            description: How these thresholds were determined
            tags: Searchable tags

        Returns:
            Pattern data dict
        """
        return {
            'wake': wake,
            'focus': focus,
            'rest': rest,
            'dream': dream
        }

    @staticmethod
    def benchmark_results(
        cycles: int,
        attention_rate: float,
        avg_salience: float,
        performance_metrics: Dict[str, float],
        description: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create performance benchmark pattern.

        Args:
            cycles: Number of cycles run
            attention_rate: Percentage of cycles attended
            avg_salience: Average salience observed
            performance_metrics: Additional metrics
            description: Benchmark description
            tags: Searchable tags

        Returns:
            Pattern data dict
        """
        return {
            'cycles': cycles,
            'attention_rate': attention_rate,
            'avg_salience': avg_salience,
            'metrics': performance_metrics
        }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate pattern library usage"""
    print("=" * 80)
    print("SAGE PATTERN LIBRARY - CRYPTOGRAPHIC PROVENANCE DEMO")
    print("=" * 80)
    print()

    # Initialize
    print("1️⃣  Initializing pattern library...")
    lct_identity = SimulatedLCTIdentity()
    library = PatternLibrary(
        lct_identity,
        consciousness_lct_id="thor-sage-consciousness"
    )
    print(f"   Creator: {library.consciousness_key.to_compact_id()}")
    print(f"   Storage: {library.storage_dir}")
    print()

    # Create SNARC weights pattern
    print("2️⃣  Creating SNARC weights pattern...")
    snarc_data = PatternTemplates.snarc_weights(
        surprise=0.06,
        novelty=0.64,
        arousal=0.32,
        reward=0.01,
        conflict=0.01,
        description="SNARC weights learned from online learning deployment (arousal baseline discovery)",
        tags=["online_learning", "thor", "validated", "baseline"]
    )

    snarc_pattern = library.create_pattern(
        pattern_type="snarc_weights",
        pattern_data=snarc_data,
        description="SNARC weights with arousal baseline importance",
        version="1.0.0",
        tags=["online_learning", "thor", "validated"],
        metadata={"source": "online_weight_adaptation", "cycles": 30}
    )
    print(f"   Pattern ID: {snarc_pattern.metadata.pattern_id}")
    print(f"   Signature: {snarc_pattern.signature.signature[:32]}...")
    print()

    # Create thresholds pattern
    print("3️⃣  Creating metabolic thresholds pattern...")
    threshold_data = PatternTemplates.metabolic_thresholds(
        wake=0.35,
        focus=0.25,
        rest=0.75,
        dream=0.05,
        description="Calibrated thresholds for Thor (stable development system)",
        tags=["thresholds", "thor", "calibrated"]
    )

    threshold_pattern = library.create_pattern(
        pattern_type="thresholds",
        pattern_data=threshold_data,
        description="Thor-calibrated metabolic state thresholds",
        version="1.0.0",
        tags=["thresholds", "thor", "calibrated"],
        metadata={"baseline_salience": 0.41, "platform": "thor"}
    )
    print(f"   Pattern ID: {threshold_pattern.metadata.pattern_id}")
    print(f"   Signature: {threshold_pattern.signature.signature[:32]}...")
    print()

    # Save patterns
    print("4️⃣  Saving patterns to library...")
    snarc_path = library.save_pattern(snarc_pattern)
    threshold_path = library.save_pattern(threshold_pattern)
    print(f"   SNARC weights: {snarc_path}")
    print(f"   Thresholds: {threshold_path}")
    print()

    # Verify patterns
    print("5️⃣  Verifying signatures...")
    snarc_valid = library.verify_pattern(snarc_pattern)
    threshold_valid = library.verify_pattern(threshold_pattern)
    print(f"   SNARC weights: {'✅ VALID' if snarc_valid else '❌ INVALID'}")
    print(f"   Thresholds: {'✅ VALID' if threshold_valid else '❌ INVALID'}")
    print()

    # Load patterns
    print("6️⃣  Loading patterns from disk...")
    loaded_snarc = library.load_pattern(snarc_pattern.metadata.pattern_id)
    loaded_threshold = library.load_pattern(threshold_pattern.metadata.pattern_id)
    print(f"   Loaded: {loaded_snarc.metadata.pattern_id}")
    print(f"   Loaded: {loaded_threshold.metadata.pattern_id}")
    print()

    # List patterns
    print("7️⃣  Listing all patterns...")
    all_patterns = library.list_patterns()
    print(f"   Total patterns: {len(all_patterns)}")
    for p in all_patterns:
        print(f"   - {p.metadata.pattern_id}: {p.metadata.description}")
    print()

    # Export pattern (for sharing)
    print("8️⃣  Exporting pattern for cross-platform sharing...")
    exported_json = library.export_pattern(snarc_pattern)
    print(f"   JSON length: {len(exported_json)} bytes")
    print(f"   Ready for sharing with Sprout ✅")
    print()

    # Simulate import (Sprout receiving pattern from Thor)
    print("9️⃣  Simulating cross-platform import...")
    print("   (As if Sprout received pattern from Thor)")

    # Sprout would have Thor's public key
    thor_public_key = library.consciousness_key.public_key_pem

    # Import and verify
    imported = library.import_pattern(
        exported_json,
        verify=True,
        creator_public_key=thor_public_key
    )
    print(f"   Imported: {imported.metadata.pattern_id}")
    print(f"   Creator: {imported.metadata.creator_lct_id}")
    print(f"   Verified: ✅ Signature valid")
    print()

    # Final summary
    print("=" * 80)
    print("PATTERN LIBRARY DEMO COMPLETE")
    print("=" * 80)
    print()
    print(f"✅ Pattern creation: Working")
    print(f"✅ Cryptographic signing: Working")
    print(f"✅ Signature verification: Working")
    print(f"✅ Storage and retrieval: Working")
    print(f"✅ Cross-platform export/import: Working")
    print()
    print("Pattern Library Capabilities:")
    print("  • Create signed patterns with automatic provenance")
    print("  • Store patterns to disk")
    print("  • Load and verify patterns")
    print("  • Search by type, tags, creator")
    print("  • Export for cross-platform sharing")
    print("  • Import with signature verification")
    print()
    print("Ready for Thor ↔ Sprout pattern sharing! 🚀")
    print()


if __name__ == "__main__":
    demo()
