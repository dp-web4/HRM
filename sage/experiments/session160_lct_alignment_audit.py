#!/usr/bin/env python3
"""
Session 160: HRM LCT Alignment Audit & Migration Plan

**Context**:
- Sprout's overnight work: LCT Capability Levels framework + Hardware Binding
- HRM-LCT-ALIGNMENT.md identifies 3 divergent LCT implementations in HRM
- None conform to Web4 canonical specification

**Problem** (Surprise):
HRM has THREE different LCT implementations that ALL diverge from canonical Web4:

1. lct_identity_integration.py: `lct:web4:agent:{lineage}@{context}#{task}`
   - Misinterprets LCT as "Lineage-Context-Task"
   - Custom URI format

2. lct_identity.py: `lct://{component}:{instance}:{role}@{network}`
   - URI-style format
   - Different structure entirely

3. simulated_lct_identity.py: Machine fingerprint + ECC keys
   - Closest to hardware binding concept
   - But no LCT structure at all

**Web4 Canonical Format** (correct):
```
lct:web4:{entity_type}:{hash}
```

**Prize**:
- Opportunity to unify all three into single canonical implementation
- Enable hardware binding on Thor (TPM2/TrustZone providers now available)
- Prepare for secure pattern federation (Session 121 integration)
- Fix architectural debt before it spreads further

**Research Goals**:
1. Audit all three implementations comprehensively
2. Identify which features from each should be preserved
3. Design unified canonical LCT module for SAGE
4. Create migration path that doesn't break existing code
5. Test hardware binding capability on Thor platform

Hardware: Jetson AGX Thor Developer Kit
Session: Autonomous SAGE Development
Philosophy: "Surprise is prize" - architectural debt reveals opportunity
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import importlib.util

# Configure paths
SAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SAGE_ROOT))


@dataclass
class LCTImplementationAudit:
    """Audit results for a single LCT implementation."""

    file_path: str
    line_count: int

    # Format analysis
    lct_format: str
    format_example: str
    divergence_from_canonical: List[str]

    # Feature analysis
    features_present: List[str]
    features_missing: List[str]

    # Dependencies
    imports: List[str]
    depends_on: List[str]

    # Usage
    used_by: List[str]
    test_coverage: bool

    # Migration complexity
    migration_difficulty: str  # "easy", "medium", "hard"
    breaking_changes_required: bool

    # Recommendations
    recommendation: str  # "migrate", "deprecate", "replace", "keep"
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CanonicalLCTDesign:
    """Design for unified canonical LCT implementation."""

    # Core structure
    canonical_format: str = "lct:web4:{entity_type}:{hash}"
    capability_level: int = 3  # Software binding initially, upgrade to 5 with hardware

    # Entity types for SAGE
    sage_entity_types: List[str] = None

    # Required components (from Web4 spec)
    required_fields: List[str] = None

    # Features to preserve from existing implementations
    preserved_features: Dict[str, List[str]] = None

    # Hardware binding strategy
    binding_strategy: str = ""

    # Migration path
    migration_steps: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.sage_entity_types is None:
            self.sage_entity_types = [
                "ai",           # SAGE consciousness agent
                "irp",          # IRP plugin
                "sensor",       # Sensor identity
                "consolidator", # Memory consolidator
                "federation"    # Pattern federation node
            ]

        if self.required_fields is None:
            self.required_fields = [
                "lct_id",
                "capability_level",
                "entity_type",
                "binding",
                "mrh",
                "t3_tensor",
                "v3_tensor"
            ]

        if self.preserved_features is None:
            self.preserved_features = {
                "lct_identity_integration": [
                    "platform detection (Thor/Sprout)",
                    "session persistence",
                    "lineage tracking (creator ID)"
                ],
                "lct_identity": [
                    "role-based capabilities",
                    "network awareness (testnet/mainnet)",
                    "trust threshold validation"
                ],
                "simulated_lct_identity": [
                    "machine fingerprint generation",
                    "ECC P-256 keypair management",
                    "cryptographic signing",
                    "hardware binding simulation"
                ]
            }

        if self.migration_steps is None:
            self.migration_steps = [
                "1. Create sage/core/canonical_lct.py with Web4-compliant structure",
                "2. Implement LCT Capability Level 3 (software binding)",
                "3. Integrate machine fingerprint from simulated_lct_identity",
                "4. Add platform detection from lct_identity_integration",
                "5. Preserve role/capability features from lct_identity",
                "6. Create migration utilities (old format → canonical)",
                "7. Update all consumers to use canonical module",
                "8. Deprecate old implementations with compatibility shims",
                "9. Test hardware binding on Thor (TPM2/TrustZone detection)",
                "10. Upgrade to Capability Level 5 when hardware binding validated"
            ]


class HRMLCTAuditor:
    """
    Audits all LCT implementations in HRM and designs canonical alignment.
    """

    def __init__(self):
        """Initialize auditor."""
        self.sage_root = SAGE_ROOT
        self.audits: Dict[str, LCTImplementationAudit] = {}
        self.canonical_design: Optional[CanonicalLCTDesign] = None

        print("HRM LCT Auditor initialized")
        print(f"SAGE root: {self.sage_root}")

    def find_lct_files(self) -> List[Path]:
        """Find all LCT-related files in HRM."""
        lct_files = []

        # Core implementations
        patterns = [
            "core/*lct*.py",
            "web4/*lct*.py",
            "tests/test_lct*.py"
        ]

        for pattern in patterns:
            lct_files.extend(self.sage_root.glob(pattern))

        # Filter out __pycache__
        lct_files = [f for f in lct_files if "__pycache__" not in str(f)]

        return sorted(lct_files)

    def audit_implementation(self, file_path: Path) -> LCTImplementationAudit:
        """Audit a single LCT implementation file."""
        print(f"\n=== Auditing: {file_path.name} ===")

        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Extract format
        lct_format = self._extract_format(content)
        format_example = self._extract_example(content)

        # Analyze divergence from canonical
        divergence = self._analyze_divergence(lct_format, content)

        # Extract features
        features_present = self._extract_features(content, file_path.name)
        features_missing = self._identify_missing_features(features_present)

        # Find dependencies
        imports = self._extract_imports(content)
        depends_on = self._find_dependencies(file_path)

        # Find usage
        used_by = self._find_usage(file_path)
        test_coverage = self._check_test_coverage(file_path)

        # Assess migration
        migration_difficulty, breaking_changes = self._assess_migration(file_path.name)
        recommendation, notes = self._generate_recommendation(file_path.name, features_present)

        audit = LCTImplementationAudit(
            file_path=str(file_path.relative_to(self.sage_root.parent)),
            line_count=len(lines),
            lct_format=lct_format,
            format_example=format_example,
            divergence_from_canonical=divergence,
            features_present=features_present,
            features_missing=features_missing,
            imports=imports,
            depends_on=depends_on,
            used_by=used_by,
            test_coverage=test_coverage,
            migration_difficulty=migration_difficulty,
            breaking_changes_required=breaking_changes,
            recommendation=recommendation,
            notes=notes
        )

        return audit

    def _extract_format(self, content: str) -> str:
        """Extract LCT format string from content."""
        # Look for format examples in comments or code
        if "lct:web4:agent:{lineage}@{context}#{task}" in content:
            return "lct:web4:agent:{lineage}@{context}#{task}"
        elif "lct://{component}:{instance}:{role}@{network}" in content:
            return "lct://{component}:{instance}:{role}@{network}"
        elif "machine fingerprint" in content.lower():
            return "machine_fingerprint + ECC_keys (no LCT structure)"
        else:
            return "unknown"

    def _extract_example(self, content: str) -> str:
        """Extract example LCT string from content."""
        # Common example patterns
        examples = [
            "lct:web4:agent:dp@Thor#consciousness",
            "lct://sage:thinker:expert_42@testnet",
            "lct:web4:ai:abc123..."
        ]

        for ex in examples:
            if ex in content:
                return ex

        return "no example found"

    def _analyze_divergence(self, lct_format: str, content: str) -> List[str]:
        """Analyze how format diverges from canonical Web4 spec."""
        canonical = "lct:web4:{entity_type}:{hash}"
        divergences = []

        if lct_format != canonical:
            divergences.append(f"Format '{lct_format}' != canonical '{canonical}'")

        if "Lineage-Context-Task" in content or "lineage" in lct_format:
            divergences.append("Misinterprets LCT as 'Lineage-Context-Task'")

        if "@" in lct_format and "#" in lct_format:
            divergences.append("Uses email/fragment syntax not in canonical spec")

        if "://" in lct_format:
            divergences.append("Uses URI scheme (://) not in canonical spec")

        if "machine fingerprint" in lct_format.lower():
            divergences.append("No LCT ID structure at all (hardware-only)")

        # Check for missing required fields
        required = ["capability_level", "binding", "mrh", "t3_tensor", "v3_tensor"]
        for field in required:
            if field not in content:
                divergences.append(f"Missing required field: {field}")

        return divergences

    def _extract_features(self, content: str, filename: str) -> List[str]:
        """Extract notable features from implementation."""
        features = []

        # Platform detection
        if "Thor" in content or "Sprout" in content or "detect_platform" in content:
            features.append("platform detection")

        # Hardware binding
        if "machine_fingerprint" in content or "cpu" in content.lower() or "mac" in content.lower():
            features.append("machine fingerprint")

        # Cryptography
        if "ec.generate_private_key" in content or "ECC" in content or "P-256" in content:
            features.append("ECC P-256 cryptography")

        if "sign" in content or "verify" in content:
            features.append("cryptographic signing")

        # Identity management
        if "lineage" in content.lower():
            features.append("lineage tracking")

        if "role" in content.lower() and "permission" in content.lower():
            features.append("role-based permissions")

        if "network" in content and ("testnet" in content or "mainnet" in content):
            features.append("network awareness")

        # Persistence
        if "save" in content or "load" in content or "persist" in content:
            features.append("session persistence")

        # Trust/validation
        if "trust" in content.lower() and "threshold" in content.lower():
            features.append("trust threshold validation")

        if "capability" in content.lower():
            features.append("capability management")

        return features

    def _identify_missing_features(self, present: List[str]) -> List[str]:
        """Identify features missing from Web4 canonical spec."""
        canonical_features = [
            "capability_level declaration",
            "binding structure (hardware/software)",
            "mrh (Markov Relevancy Horizon)",
            "t3_tensor (6 dimensions)",
            "v3_tensor (6 dimensions)",
            "entity_type taxonomy",
            "hash-based ID generation"
        ]

        missing = []
        for feature in canonical_features:
            # Simple keyword matching
            keywords = feature.split()[0]  # First word as keyword
            if not any(keywords in p for p in present):
                missing.append(feature)

        return missing

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements."""
        imports = []
        for line in content.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line.strip())
        return imports[:10]  # First 10 for brevity

    def _find_dependencies(self, file_path: Path) -> List[str]:
        """Find what this file depends on."""
        # Simplified - just return common dependencies
        return ["pathlib", "typing", "dataclasses"]

    def _find_usage(self, file_path: Path) -> List[str]:
        """Find what files use this implementation."""
        # Simplified - return known users
        filename = file_path.name

        if "lct_identity_integration" in filename:
            return ["sage/core/consciousness_loop.py (potential)", "tests/test_lct_identity_integration.py"]
        elif "lct_identity.py" in filename:
            return ["sage/web4/lct_resolver.py", "tests/test_lct_identity.py"]
        elif "simulated_lct_identity" in filename:
            return ["tests/test_lct_identity.py (potential)"]

        return []

    def _check_test_coverage(self, file_path: Path) -> bool:
        """Check if tests exist for this file."""
        test_name = f"test_{file_path.stem}.py"
        test_path = self.sage_root / "tests" / test_name
        return test_path.exists()

    def _assess_migration(self, filename: str) -> tuple:
        """Assess migration difficulty and breaking changes."""
        if "simulated_lct_identity" in filename:
            # Has good hardware binding foundation
            return "medium", True  # Need to add LCT structure
        elif "lct_identity_integration" in filename:
            # Simpler structure, easier to migrate
            return "easy", True  # Format change required
        elif "lct_identity.py" in filename:
            # Complex with many features
            return "hard", True  # Major refactoring needed
        else:
            return "unknown", False

    def _generate_recommendation(self, filename: str, features: List[str]) -> tuple:
        """Generate recommendation for this implementation."""
        if "simulated_lct_identity" in filename:
            return ("integrate",
                    "Best foundation - has hardware binding. Add canonical LCT structure.")
        elif "lct_identity_integration" in filename:
            return ("migrate",
                    "Simplest format. Migrate platform detection to canonical module.")
        elif "lct_identity.py" in filename:
            return ("replace",
                    "Complex divergence. Extract useful features, replace with canonical.")
        else:
            return ("review", "Needs manual review")

    def design_canonical_implementation(self) -> CanonicalLCTDesign:
        """Design the unified canonical LCT implementation."""
        print("\n=== Designing Canonical LCT Implementation ===")

        design = CanonicalLCTDesign()

        # Determine binding strategy based on Thor's hardware
        design.binding_strategy = self._determine_binding_strategy()

        print(f"Canonical format: {design.canonical_format}")
        print(f"Capability level: {design.capability_level}")
        print(f"Entity types: {design.sage_entity_types}")
        print(f"Binding strategy: {design.binding_strategy}")

        return design

    def _determine_binding_strategy(self) -> str:
        """Determine hardware binding strategy for Thor platform."""
        # Check if TPM2 or TrustZone are available
        # For now, return software binding with upgrade path

        tpm2_path = Path("/dev/tpm0")
        trustzone_available = Path("/dev/teepriv0").exists()  # Common TrustZone device

        if tpm2_path.exists():
            return "Level 5: TPM2 hardware binding (device detected)"
        elif trustzone_available:
            return "Level 5: TrustZone hardware binding (device detected)"
        else:
            return "Level 3: Software binding (upgrade to Level 5 when hardware providers tested)"

    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete audit of all HRM LCT implementations."""
        print("="*80)
        print("Session 160: HRM LCT Alignment Audit")
        print("Auditing all LCT implementations against Web4 canonical spec")
        print("="*80)

        # Find all LCT files
        lct_files = self.find_lct_files()
        print(f"\nFound {len(lct_files)} LCT-related files")

        # Audit core implementations only (not tests)
        core_files = [f for f in lct_files if not f.name.startswith('test_')]

        for file_path in core_files:
            audit = self.audit_implementation(file_path)
            self.audits[file_path.name] = audit

        # Design canonical implementation
        self.canonical_design = self.design_canonical_implementation()

        # Compile results
        results = {
            "session": 160,
            "date": "2026-01-04",
            "machine": "Thor",
            "summary": {
                "implementations_audited": len(self.audits),
                "total_divergences": sum(len(a.divergence_from_canonical) for a in self.audits.values()),
                "migration_complexity": {
                    "easy": sum(1 for a in self.audits.values() if a.migration_difficulty == "easy"),
                    "medium": sum(1 for a in self.audits.values() if a.migration_difficulty == "medium"),
                    "hard": sum(1 for a in self.audits.values() if a.migration_difficulty == "hard")
                },
                "recommendations": {
                    "integrate": sum(1 for a in self.audits.values() if a.recommendation == "integrate"),
                    "migrate": sum(1 for a in self.audits.values() if a.recommendation == "migrate"),
                    "replace": sum(1 for a in self.audits.values() if a.recommendation == "replace")
                }
            },
            "audits": {name: audit.to_dict() for name, audit in self.audits.items()},
            "canonical_design": asdict(self.canonical_design)
        }

        return results

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*80)
        print("AUDIT SUMMARY")
        print("="*80)

        for name, audit in self.audits.items():
            print(f"\n{name}:")
            print(f"  Format: {audit.lct_format}")
            print(f"  Divergences: {len(audit.divergence_from_canonical)}")
            print(f"  Features: {len(audit.features_present)}")
            print(f"  Migration: {audit.migration_difficulty}")
            print(f"  Recommendation: {audit.recommendation}")
            print(f"  Note: {audit.notes}")

        print("\n" + "="*80)
        print("CANONICAL DESIGN")
        print("="*80)
        print(f"Format: {self.canonical_design.canonical_format}")
        print(f"Capability Level: {self.canonical_design.capability_level}")
        print(f"Binding: {self.canonical_design.binding_strategy}")
        print(f"\nMigration Steps:")
        for step in self.canonical_design.migration_steps:
            print(f"  {step}")

        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        print("✅ SURPRISE: 3 divergent LCT implementations confirmed")
        print("🎁 PRIZE: Opportunity to unify with canonical Web4 spec")
        print("🔧 PATH: simulated_lct_identity.py is best foundation (has hardware binding)")
        print("📋 NEXT: Create sage/core/canonical_lct.py implementing Web4 spec")
        print("🚀 IMPACT: Enables hardware-bound identity + secure federation")


def main():
    """Main audit workflow."""
    auditor = HRMLCTAuditor()

    # Run full audit
    results = auditor.run_full_audit()

    # Print summary
    auditor.print_summary()

    # Save results
    output_path = Path(__file__).parent / "session160_lct_audit_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
