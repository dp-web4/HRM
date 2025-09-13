"""
SPARQL Query Examples for MRH Graph Traversal
==============================================

Demonstrates practical queries for navigating the Markov Relevancy Horizon
as RDF graphs within the Web4 standard.
"""

from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD
import numpy as np


# Define namespaces
MRH = Namespace("https://web4.foundation/mrh/v1#")
LCT = Namespace("https://web4.foundation/lct/")
WEB4 = Namespace("https://web4.foundation/ontology#")


class MRHQueryEngine:
    """Query engine for MRH graph traversal"""
    
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("mrh", MRH)
        self.graph.bind("lct", LCT)
        self.graph.bind("web4", WEB4)
        self.graph.bind("xsd", XSD)
    
    def load_lct_mrh(self, lct_data: Dict[str, Any]):
        """Load an LCT's MRH graph into the query engine"""
        if "@graph" in lct_data.get("mrh", {}):
            # Parse JSON-LD format
            self.graph.parse(data=json.dumps(lct_data["mrh"]), format="json-ld")
    
    # ============================================
    # Core Navigation Queries
    # ============================================
    
    def find_high_probability_paths(self, min_probability: float = 0.7) -> List[Dict]:
        """Find all high-probability relevance paths"""
        query = """
        SELECT ?target ?probability ?relation ?distance
        WHERE {
            ?relevance a mrh:Relevance ;
                      mrh:target ?target ;
                      mrh:probability ?probability ;
                      mrh:relation ?relation .
            OPTIONAL { ?relevance mrh:distance ?distance }
            FILTER(?probability >= %f)
        }
        ORDER BY DESC(?probability)
        """ % min_probability
        
        results = []
        for row in self.graph.query(query):
            results.append({
                "target": str(row.target),
                "probability": float(row.probability),
                "relation": str(row.relation),
                "distance": int(row.distance) if row.distance else 1
            })
        return results
    
    def compute_reachability(self, max_distance: int = 3) -> List[Dict]:
        """Compute all reachable LCTs within distance threshold"""
        query = """
        SELECT ?target ?prob ?dist
        WHERE {
            ?rel mrh:target ?target ;
                 mrh:probability ?prob .
            OPTIONAL { ?rel mrh:distance ?dist }
            FILTER(!BOUND(?dist) || ?dist <= %d)
        }
        ORDER BY DESC(?prob)
        """ % max_distance
        
        results = []
        for row in self.graph.query(query):
            dist = int(row.dist) if row.dist else 1
            if dist <= max_distance:
                # Compute decay manually
                decay = 0.9 ** (dist - 1)
                total_prob = float(row.prob) * decay
                results.append({
                    "target": str(row.target),
                    "reachability": total_prob
                })
        return results
    
    def find_conditional_paths(self) -> List[Dict]:
        """Find all conditional dependencies in the graph"""
        query = """
        SELECT ?target ?probability ?condition
        WHERE {
            ?rel mrh:target ?target ;
                 mrh:probability ?probability ;
                 mrh:conditional_on ?condition .
        }
        """
        
        results = []
        for row in self.graph.query(query):
            results.append({
                "target": str(row.target),
                "probability": float(row.probability),
                "condition": str(row.condition)
            })
        return results
    
    def find_alternative_solutions(self) -> List[List[Dict]]:
        """Find mutually exclusive alternative paths"""
        query = """
        SELECT ?group ?target ?probability
        WHERE {
            ?rel mrh:target ?target ;
                 mrh:probability ?probability ;
                 mrh:relation mrh:alternatives_to ;
                 mrh:alternatives ?group .
        }
        ORDER BY ?group DESC(?probability)
        """
        
        alternatives = {}
        for row in self.graph.query(query):
            group = str(row.group)
            if group not in alternatives:
                alternatives[group] = []
            alternatives[group].append({
                "target": str(row.target),
                "probability": float(row.probability)
            })
        
        return list(alternatives.values())
    
    # ============================================
    # Trust Propagation Queries
    # ============================================
    
    def calculate_trust_flow(self, source_trust: float = 1.0) -> Dict[str, float]:
        """Calculate trust propagation through the graph"""
        query = """
        SELECT ?target ?probability ?distance ?trust
        WHERE {
            ?rel mrh:target ?target ;
                 mrh:probability ?probability .
            OPTIONAL { ?rel mrh:distance ?distance }
            OPTIONAL { ?rel mrh:trust ?trust }
        }
        """
        
        trust_scores = {}
        for row in self.graph.query(query):
            target = str(row.target)
            prob = float(row.probability)
            dist = int(row.distance) if row.distance else 1
            edge_trust = float(row.trust) if row.trust else 1.0
            
            # Trust = source_trust * probability * edge_trust * decay
            decay = 0.9 ** (dist - 1)
            trust = source_trust * prob * edge_trust * decay
            
            # Accumulate trust if multiple paths
            if target in trust_scores:
                # Combine trust using probability OR
                trust_scores[target] = trust_scores[target] + trust - (trust_scores[target] * trust)
            else:
                trust_scores[target] = trust
        
        return trust_scores
    
    # ============================================
    # Semantic Clustering Queries
    # ============================================
    
    def find_semantic_clusters(self) -> List[List[str]]:
        """Identify semantic clusters based on relations"""
        query = """
        SELECT ?rel1 ?rel2 ?target1 ?target2
        WHERE {
            ?rel1 mrh:target ?target1 ;
                  mrh:relation ?relation .
            ?rel2 mrh:target ?target2 ;
                  mrh:relation ?relation .
            FILTER(?rel1 != ?rel2)
        }
        """
        
        # Build adjacency for clustering
        clusters = {}
        for row in self.graph.query(query):
            t1, t2 = str(row.target1), str(row.target2)
            for t in [t1, t2]:
                if t not in clusters:
                    clusters[t] = set()
            clusters[t1].add(t2)
            clusters[t2].add(t1)
        
        # Extract connected components
        visited = set()
        components = []
        for node in clusters:
            if node not in visited:
                component = []
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(clusters.get(current, set()))
                if component:
                    components.append(component)
        
        return components
    
    # ============================================
    # Temporal Decay Queries
    # ============================================
    
    def apply_temporal_decay(self, current_time: int) -> Dict[str, float]:
        """Apply temporal decay to relevance scores"""
        query = """
        SELECT ?target ?probability ?timestamp ?decay_rate
        WHERE {
            ?rel mrh:target ?target ;
                 mrh:probability ?probability .
            OPTIONAL { ?rel mrh:timestamp ?timestamp }
            OPTIONAL { ?rel mrh:decay_rate ?decay_rate }
        }
        """
        
        decayed_scores = {}
        for row in self.graph.query(query):
            target = str(row.target)
            prob = float(row.probability)
            timestamp = int(row.timestamp) if row.timestamp else current_time
            decay_rate = float(row.decay_rate) if row.decay_rate else 0.95
            
            # Apply exponential decay
            time_delta = current_time - timestamp
            decayed_prob = prob * (decay_rate ** time_delta)
            
            decayed_scores[target] = decayed_prob
        
        return decayed_scores
    
    # ============================================
    # Complex Reasoning Queries
    # ============================================
    
    def find_reasoning_chains(self, goal_lct: str, max_depth: int = 5) -> List[List[str]]:
        """Find reasoning chains that lead to a goal LCT"""
        # This would require recursive SPARQL or property paths
        # Simplified version using programmatic traversal
        
        chains = []
        
        def dfs(current_chain: List[str], remaining_depth: int):
            if remaining_depth == 0:
                return
            
            # Query for LCTs that reference the last in chain
            query = """
            SELECT ?source ?probability
            WHERE {
                ?rel mrh:target <%s> ;
                     mrh:probability ?probability .
                ?source_lct mrh:contains ?rel .
            }
            ORDER BY DESC(?probability)
            """ % current_chain[-1]
            
            for row in self.graph.query(query):
                source = str(row.source)
                if source not in current_chain:  # Avoid cycles
                    new_chain = current_chain + [source]
                    if source == goal_lct:
                        chains.append(new_chain)
                    else:
                        dfs(new_chain, remaining_depth - 1)
        
        # Start from current LCT context
        dfs(["current_lct"], max_depth)
        return chains
    
    def compute_joint_probability(self, path: List[str]) -> float:
        """Compute joint probability of a path through the graph"""
        if len(path) < 2:
            return 1.0
        
        joint_prob = 1.0
        for i in range(len(path) - 1):
            query = """
            SELECT ?probability
            WHERE {
                ?rel mrh:target <%s> ;
                     mrh:probability ?probability .
                FILTER(EXISTS { ?rel mrh:source <%s> })
            }
            """ % (path[i+1], path[i])
            
            results = list(self.graph.query(query))
            if results:
                joint_prob *= float(results[0].probability)
            else:
                joint_prob *= 0.1  # Default low probability for missing edges
        
        return joint_prob


def demonstrate_queries():
    """Demonstrate various SPARQL queries on MRH graphs"""
    
    # Create example LCT with complex MRH
    example_lct = {
        "lct_version": "1.0",
        "entity_id": "entity:research_paper",
        "mrh": {
            "@context": {
                "@vocab": "https://web4.foundation/mrh/v1#",
                "mrh": "https://web4.foundation/mrh/v1#",
                "lct": "https://web4.foundation/lct/",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": [
                {
                    "@id": "_:ref1",
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:foundation_paper"},
                    "mrh:probability": {"@value": "0.95", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:derives_from",
                    "mrh:distance": {"@value": "1", "@type": "xsd:integer"},
                    "mrh:timestamp": {"@value": "1000", "@type": "xsd:integer"}
                },
                {
                    "@id": "_:ref2",
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:related_work_1"},
                    "mrh:probability": {"@value": "0.8", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:references",
                    "mrh:distance": {"@value": "2", "@type": "xsd:integer"}
                },
                {
                    "@id": "_:ref3",
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:alternative_approach"},
                    "mrh:probability": {"@value": "0.6", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:alternatives_to",
                    "mrh:alternatives": {"@id": "_:ref4"}
                },
                {
                    "@id": "_:ref4",
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:our_approach"},
                    "mrh:probability": {"@value": "0.4", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:alternatives_to",
                    "mrh:alternatives": {"@id": "_:ref3"}
                },
                {
                    "@id": "_:ref5",
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:future_work"},
                    "mrh:probability": {"@value": "0.7", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:extends",
                    "mrh:conditional_on": {"@id": "_:ref1"},
                    "mrh:distance": {"@value": "3", "@type": "xsd:integer"}
                }
            ]
        }
    }
    
    # Initialize query engine
    engine = MRHQueryEngine()
    engine.load_lct_mrh(example_lct)
    
    print("=" * 60)
    print("MRH SPARQL Query Demonstrations")
    print("=" * 60)
    
    # 1. High probability paths
    print("\n1. HIGH PROBABILITY PATHS (p >= 0.7):")
    print("-" * 40)
    paths = engine.find_high_probability_paths(0.7)
    for path in paths:
        target = path['target'].split('/')[-1] if '/' in path['target'] else path['target'].split(':')[-1]
        relation = path['relation'].split('#')[-1] if '#' in path['relation'] else path['relation'].split('/')[-1]
        print(f"  → {target}")
        print(f"    Probability: {path['probability']:.2f}")
        print(f"    Relation: {relation}")
        print(f"    Distance: {path['distance']}")
    
    # 2. Reachability computation
    print("\n2. REACHABILITY WITHIN 3 HOPS:")
    print("-" * 40)
    reachable = engine.compute_reachability(3)
    for node in reachable:
        target = node['target'].split('/')[-1] if '/' in node['target'] else node['target'].split(':')[-1]
        print(f"  → {target}: {node['reachability']:.3f}")
    
    # 3. Trust propagation
    print("\n3. TRUST PROPAGATION:")
    print("-" * 40)
    trust = engine.calculate_trust_flow(source_trust=0.9)
    for target_uri, score in sorted(trust.items(), key=lambda x: x[1], reverse=True):
        target = target_uri.split('/')[-1] if '/' in target_uri else target_uri.split(':')[-1]
        print(f"  → {target}: {score:.3f}")
    
    # 4. Alternative solutions
    print("\n4. ALTERNATIVE SOLUTIONS:")
    print("-" * 40)
    alternatives = engine.find_alternative_solutions()
    for i, group in enumerate(alternatives, 1):
        print(f"  Group {i}:")
        for alt in group:
            target = alt['target'].split('/')[-1] if '/' in alt['target'] else alt['target'].split(':')[-1]
            print(f"    → {target}: p={alt['probability']:.2f}")
    
    # 5. Conditional dependencies
    print("\n5. CONDITIONAL DEPENDENCIES:")
    print("-" * 40)
    conditionals = engine.find_conditional_paths()
    for cond in conditionals:
        target = cond['target'].split('/')[-1] if '/' in cond['target'] else cond['target'].split(':')[-1]
        condition = cond['condition'].split('/')[-1] if '/' in cond['condition'] else cond['condition'].split(':')[-1]
        print(f"  → {target} depends on {condition}")
        print(f"    Joint probability: {cond['probability']:.2f}")
    
    # 6. Temporal decay
    print("\n6. TEMPORAL DECAY (current_time=2000):")
    print("-" * 40)
    decayed = engine.apply_temporal_decay(current_time=2000)
    for target_uri, score in sorted(decayed.items(), key=lambda x: x[1], reverse=True):
        target = target_uri.split('/')[-1] if '/' in target_uri else target_uri.split(':')[-1]
        print(f"  → {target}: {score:.3f}")
    
    print("\n" + "=" * 60)
    print("Query demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_queries()