"""
MRH Migration Tool: Simple Array to RDF Graph
=============================================

Migrates existing Web4 LCTs from simple MRH arrays to rich RDF graphs,
with intelligent inference of relationships and probabilities.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MigrationStats:
    """Statistics for migration process"""
    total_lcts: int = 0
    migrated: int = 0
    failed: int = 0
    already_rdf: int = 0
    simple_arrays: int = 0
    inferred_relations: int = 0
    

class MRHMigrator:
    """
    Migrates MRH from simple array format to RDF graph format.
    
    Features:
    - Automatic relationship inference
    - Probability estimation based on order
    - Distance calculation
    - Backward compatibility preservation
    """
    
    def __init__(self, 
                 infer_relations: bool = True,
                 preserve_original: bool = True,
                 default_probability_decay: float = 0.9):
        self.infer_relations = infer_relations
        self.preserve_original = preserve_original
        self.default_probability_decay = default_probability_decay
        self.stats = MigrationStats()
        
        # Relation inference patterns
        self.relation_patterns = {
            "parent": "mrh:derives_from",
            "child": "mrh:extends",
            "sibling": "mrh:references",
            "prev": "mrh:derives_from",
            "next": "mrh:extends",
            "related": "mrh:references",
            "source": "mrh:derives_from",
            "target": "mrh:produces",
            "input": "mrh:depends_on",
            "output": "mrh:produces",
            "alternative": "mrh:alternatives_to",
            "contradiction": "mrh:contradicts",
            "specialization": "mrh:specializes"
        }
    
    def migrate_lct(self, lct_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a single LCT from simple to RDF format.
        Returns migrated LCT or original if already RDF.
        """
        self.stats.total_lcts += 1
        
        # Check if MRH exists
        if "mrh" not in lct_data:
            return lct_data
        
        mrh = lct_data["mrh"]
        
        # Check if already RDF format
        if isinstance(mrh, dict) and "@graph" in mrh:
            self.stats.already_rdf += 1
            return lct_data
        
        # Check if simple array format
        if not isinstance(mrh, list):
            self.stats.failed += 1
            return lct_data
        
        self.stats.simple_arrays += 1
        
        # Preserve original if requested
        if self.preserve_original:
            lct_data["mrh_original"] = mrh.copy()
        
        # Convert to RDF format
        rdf_mrh = self._convert_to_rdf(mrh, lct_data)
        lct_data["mrh"] = rdf_mrh
        
        self.stats.migrated += 1
        return lct_data
    
    def _convert_to_rdf(self, mrh_array: List[str], lct_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simple MRH array to RDF graph"""
        
        # Create RDF structure
        rdf_mrh = {
            "@context": {
                "@vocab": "https://web4.foundation/mrh/v1#",
                "mrh": "https://web4.foundation/mrh/v1#",
                "lct": "https://web4.foundation/lct/",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": []
        }
        
        # Process each reference
        for i, ref in enumerate(mrh_array):
            relevance = self._create_relevance(ref, i, mrh_array, lct_data)
            rdf_mrh["@graph"].append(relevance)
        
        # Add migration metadata
        rdf_mrh["mrh:migration"] = {
            "timestamp": datetime.now().isoformat(),
            "tool_version": "1.0",
            "original_count": len(mrh_array)
        }
        
        return rdf_mrh
    
    def _create_relevance(self, ref: str, index: int, mrh_array: List[str], 
                         lct_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a relevance entry from a simple reference"""
        
        # Base relevance structure
        relevance = {
            "@id": f"_:ref{index}",
            "@type": "mrh:Relevance",
            "mrh:target": {"@id": self._normalize_lct_uri(ref)}
        }
        
        # Calculate probability based on position
        # Earlier references get higher probability
        base_prob = self.default_probability_decay ** index
        relevance["mrh:probability"] = {
            "@value": str(round(base_prob, 3)),
            "@type": "xsd:decimal"
        }
        
        # Calculate distance
        # Assume geometric progression for distance
        if index == 0:
            distance = 1
        elif index < 3:
            distance = 2
        else:
            distance = 3
        
        relevance["mrh:distance"] = {
            "@value": str(distance),
            "@type": "xsd:integer"
        }
        
        # Infer relationship if possible
        if self.infer_relations:
            relation = self._infer_relation(ref, index, len(mrh_array), lct_data)
            relevance["mrh:relation"] = relation
            if relation != "mrh:references":
                self.stats.inferred_relations += 1
        else:
            relevance["mrh:relation"] = "mrh:references"
        
        # Add timestamp (migration time)
        relevance["mrh:timestamp"] = {
            "@value": str(int(datetime.now().timestamp())),
            "@type": "xsd:integer"
        }
        
        # Check for alternatives (consecutive similar refs)
        if index > 0 and self._are_alternatives(ref, mrh_array[index-1]):
            relevance["mrh:alternatives"] = {"@id": f"_:ref{index-1}"}
        
        return relevance
    
    def _normalize_lct_uri(self, ref: str) -> str:
        """Normalize reference to proper LCT URI format"""
        if ref.startswith("lct:"):
            return ref
        elif ref.startswith("http"):
            return ref
        else:
            # Assume it's a hash or ID
            return f"lct:{ref}"
    
    def _infer_relation(self, ref: str, index: int, total: int, 
                       lct_data: Dict[str, Any]) -> str:
        """
        Infer relationship type from reference and context.
        Uses heuristics based on naming patterns and position.
        """
        
        # Check reference naming patterns
        ref_lower = ref.lower()
        
        # Direct pattern matching
        for pattern, relation in self.relation_patterns.items():
            if pattern in ref_lower:
                return relation
        
        # Positional heuristics
        if index == 0:
            # First reference often derives from
            if "source" in ref_lower or "origin" in ref_lower:
                return "mrh:derives_from"
            elif "prev" in ref_lower or "parent" in ref_lower:
                return "mrh:derives_from"
        
        if index == total - 1:
            # Last reference often extends or produces
            if "output" in ref_lower or "result" in ref_lower:
                return "mrh:produces"
            elif "next" in ref_lower or "future" in ref_lower:
                return "mrh:extends"
        
        # Check entity metadata if available
        if "entity_type" in lct_data:
            entity_type = lct_data["entity_type"]
            if entity_type == "reasoning":
                # Reasoning chains have specific patterns
                if index == 0:
                    return "mrh:derives_from"
                elif index < total - 1:
                    return "mrh:depends_on"
                else:
                    return "mrh:produces"
            elif entity_type == "document":
                return "mrh:references"
            elif entity_type == "contradiction":
                return "mrh:contradicts"
        
        # Default to references
        return "mrh:references"
    
    def _are_alternatives(self, ref1: str, ref2: str) -> bool:
        """Check if two references are likely alternatives"""
        # Simple heuristic: similar prefixes or patterns
        if ref1[:10] == ref2[:10]:  # Same prefix
            return True
        if "alternative" in ref1.lower() and "alternative" in ref2.lower():
            return True
        if "option" in ref1.lower() and "option" in ref2.lower():
            return True
        return False
    
    def migrate_batch(self, lcts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Migrate a batch of LCTs"""
        migrated = []
        for lct in lcts:
            migrated.append(self.migrate_lct(lct))
        return migrated
    
    def migrate_file(self, input_path: str, output_path: str) -> MigrationStats:
        """Migrate LCTs from input file to output file"""
        with open(input_path, 'r') as f:
            if input_path.endswith('.jsonl'):
                # JSONL format (one LCT per line)
                lcts = [json.loads(line) for line in f]
            else:
                # JSON array format
                lcts = json.load(f)
                if not isinstance(lcts, list):
                    lcts = [lcts]
        
        # Migrate all LCTs
        migrated = self.migrate_batch(lcts)
        
        # Write output
        with open(output_path, 'w') as f:
            if output_path.endswith('.jsonl'):
                for lct in migrated:
                    f.write(json.dumps(lct) + '\n')
            else:
                json.dump(migrated, f, indent=2)
        
        return self.stats
    
    def validate_migration(self, original: Dict[str, Any], 
                          migrated: Dict[str, Any]) -> bool:
        """Validate that migration preserves essential information"""
        # Check all original refs are present
        if "mrh" not in original or "mrh" not in migrated:
            return False
        
        original_mrh = original["mrh"]
        migrated_mrh = migrated["mrh"]
        
        if isinstance(original_mrh, list) and isinstance(migrated_mrh, dict):
            # Extract targets from migrated
            migrated_targets = []
            for relevance in migrated_mrh.get("@graph", []):
                target = relevance.get("mrh:target", {})
                if isinstance(target, dict):
                    target_id = target.get("@id", "")
                else:
                    target_id = str(target)
                
                # Normalize for comparison
                target_id = target_id.replace("lct:", "")
                migrated_targets.append(target_id)
            
            # Check all original refs are present
            for ref in original_mrh:
                ref_normalized = ref.replace("lct:", "")
                if ref_normalized not in migrated_targets:
                    return False
            
            return True
        
        return False


def demonstrate_migration():
    """Demonstrate MRH migration tool"""
    
    print("=" * 60)
    print("MRH Migration Tool Demonstration")
    print("=" * 60)
    
    # Example LCTs with simple MRH arrays
    example_lcts = [
        {
            "lct_version": "1.0",
            "entity_id": "entity:document_123",
            "entity_type": "document",
            "mrh": [
                "source_document",
                "related_paper_1",
                "related_paper_2",
                "citation_3"
            ]
        },
        {
            "lct_version": "1.0",
            "entity_id": "entity:reasoning_456",
            "entity_type": "reasoning",
            "mrh": [
                "premise_1",
                "premise_2",
                "inference_step",
                "conclusion"
            ]
        },
        {
            "lct_version": "1.0",
            "entity_id": "entity:alternatives_789",
            "mrh": [
                "base_approach",
                "alternative_method_1",
                "alternative_method_2",
                "alternative_method_3"
            ]
        }
    ]
    
    # Create migrator
    migrator = MRHMigrator(
        infer_relations=True,
        preserve_original=True,
        default_probability_decay=0.85
    )
    
    # Migrate each LCT
    for i, lct in enumerate(example_lcts, 1):
        print(f"\n{'='*40}")
        print(f"LCT {i}: {lct['entity_id']}")
        print(f"{'='*40}")
        
        print("\nORIGINAL MRH:")
        print(json.dumps(lct["mrh"], indent=2))
        
        # Migrate
        migrated = migrator.migrate_lct(lct.copy())
        
        print("\nMIGRATED MRH:")
        if "@graph" in migrated["mrh"]:
            for relevance in migrated["mrh"]["@graph"]:
                target = relevance["mrh:target"]["@id"].split(":")[-1]
                prob = relevance["mrh:probability"]["@value"]
                relation = relevance["mrh:relation"].split(":")[-1]
                dist = relevance["mrh:distance"]["@value"]
                
                print(f"  → {target}")
                print(f"    Relation: {relation}")
                print(f"    Probability: {prob}")
                print(f"    Distance: {dist}")
        
        # Validate
        is_valid = migrator.validate_migration(lct, migrated)
        print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("MIGRATION STATISTICS:")
    print("-" * 40)
    print(f"Total LCTs: {migrator.stats.total_lcts}")
    print(f"Migrated: {migrator.stats.migrated}")
    print(f"Already RDF: {migrator.stats.already_rdf}")
    print(f"Simple Arrays: {migrator.stats.simple_arrays}")
    print(f"Inferred Relations: {migrator.stats.inferred_relations}")
    print(f"Failed: {migrator.stats.failed}")
    
    # Test batch migration
    print("\n" + "=" * 60)
    print("BATCH MIGRATION TEST:")
    print("-" * 40)
    
    # Save test data
    test_input = "test_lcts_simple.json"
    test_output = "test_lcts_rdf.json"
    
    with open(test_input, 'w') as f:
        json.dump(example_lcts, f, indent=2)
    
    # Migrate file
    new_migrator = MRHMigrator()
    stats = new_migrator.migrate_file(test_input, test_output)
    
    print(f"Migrated {stats.migrated} LCTs from {test_input} to {test_output}")
    
    # Load and show one migrated example
    with open(test_output, 'r') as f:
        migrated_lcts = json.load(f)
    
    if migrated_lcts:
        print("\nFirst migrated LCT preview:")
        print(json.dumps(migrated_lcts[0]["mrh"]["@graph"][:2], indent=2))
    
    print("\n" + "=" * 60)
    print("Migration demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_migration()