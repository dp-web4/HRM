#!/usr/bin/env python3
"""
MRH RDF Implementation - Practical example of Markov Relevancy Horizon as RDF graphs
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, XSD
import networkx as nx
import matplotlib.pyplot as plt

# Define namespaces
MRH = Namespace("https://web4.foundation/mrh/v1#")
LCT = Namespace("https://web4.foundation/lct/")
WEB4 = Namespace("https://web4.foundation/web4/v1#")

class MRHRelation(Enum):
    """Standard MRH relationship types"""
    DERIVES_FROM = "derives_from"
    SPECIALIZES = "specializes"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    ALTERNATIVES_TO = "alternatives_to"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    TRANSFORMS = "transforms"

@dataclass
class MRHEdge:
    """Represents an edge in the MRH graph"""
    target_lct: str
    probability: float
    relation: MRHRelation
    distance: int = 1
    decay_rate: float = 0.9
    conditional_on: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)
    
    def effective_probability(self, current_distance: int = 0) -> float:
        """Calculate effective probability with decay"""
        return self.probability * (self.decay_rate ** (current_distance + self.distance - 1))

class MRHGraph:
    """Markov Relevancy Horizon graph implementation"""
    
    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.graph = Graph()
        self.edges: List[MRHEdge] = []
        self._setup_namespaces()
        
    def _setup_namespaces(self):
        """Initialize RDF namespaces"""
        self.graph.bind("mrh", MRH)
        self.graph.bind("lct", LCT)
        self.graph.bind("web4", WEB4)
        self.graph.bind("xsd", XSD)
        
    def add_relevance(self, edge: MRHEdge) -> BNode:
        """Add a relevance relationship to the graph"""
        relevance_node = BNode()
        
        # Add type
        self.graph.add((relevance_node, RDF.type, MRH.Relevance))
        
        # Add target
        target_uri = LCT[edge.target_lct]
        self.graph.add((relevance_node, MRH.target, target_uri))
        
        # Add probability
        self.graph.add((relevance_node, MRH.probability, 
                       Literal(edge.probability, datatype=XSD.decimal)))
        
        # Add relation
        rel_uri = MRH[edge.relation.value]
        self.graph.add((relevance_node, MRH.relation, rel_uri))
        
        # Add distance
        self.graph.add((relevance_node, MRH.distance, 
                       Literal(edge.distance, datatype=XSD.integer)))
        
        # Add decay rate
        self.graph.add((relevance_node, MRH.decay_rate,
                       Literal(edge.decay_rate, datatype=XSD.decimal)))
        
        # Add conditional dependencies
        if edge.conditional_on:
            for condition in edge.conditional_on:
                self.graph.add((relevance_node, MRH.conditional_on, LCT[condition]))
        
        # Store edge for traversal
        self.edges.append(edge)
        
        return relevance_node
    
    def to_jsonld(self) -> Dict:
        """Convert to JSON-LD format for LCT embedding"""
        # Serialize to JSON-LD
        jsonld_str = self.graph.serialize(format='json-ld')
        jsonld_dict = json.loads(jsonld_str)
        
        # Add context
        context = {
            "@vocab": "https://web4.foundation/mrh/v1#",
            "mrh": "https://web4.foundation/mrh/v1#",
            "lct": "https://web4.foundation/lct/",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }
        
        return {
            "@context": context,
            "@graph": jsonld_dict.get("@graph", [])
        }
    
    def traverse_markov(self, max_depth: int = 3, 
                       min_probability: float = 0.1) -> List[Tuple[str, float, List[str]]]:
        """
        Traverse the MRH graph using Markovian walk
        Returns: List of (lct_id, cumulative_probability, path)
        """
        results = []
        visited = set()
        
        def _traverse(current_edges: List[MRHEdge], 
                     depth: int, 
                     cum_prob: float,
                     path: List[str]):
            
            if depth >= max_depth:
                return
            
            for edge in current_edges:
                # Calculate new probability
                new_prob = cum_prob * edge.effective_probability(depth)
                
                # Skip if below threshold
                if new_prob < min_probability:
                    continue
                
                # Skip if already visited with higher probability
                if edge.target_lct in visited:
                    continue
                
                visited.add(edge.target_lct)
                new_path = path + [edge.target_lct]
                results.append((edge.target_lct, new_prob, new_path))
                
                # Recursively traverse if we can fetch the target LCT
                target_mrh = self._fetch_lct_mrh(edge.target_lct)
                if target_mrh:
                    _traverse(target_mrh.edges, depth + 1, new_prob, new_path)
        
        # Start traversal from root edges
        _traverse(self.edges, 0, 1.0, [])
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _fetch_lct_mrh(self, lct_id: str) -> Optional['MRHGraph']:
        """Fetch MRH graph from another LCT (simulated)"""
        # In real implementation, this would fetch from network
        # For demo, we'll return None or generate synthetic data
        return None
    
    def find_paths(self, target_lct: str, max_paths: int = 5) -> List[List[MRHEdge]]:
        """Find multiple paths to a target LCT"""
        paths = []
        
        def _dfs(current_edges: List[MRHEdge], 
                target: str, 
                current_path: List[MRHEdge],
                visited: Set[str]):
            
            if len(paths) >= max_paths:
                return
            
            for edge in current_edges:
                if edge.target_lct == target:
                    paths.append(current_path + [edge])
                elif edge.target_lct not in visited:
                    visited.add(edge.target_lct)
                    # Would recursively search in real implementation
                    _dfs([], target, current_path + [edge], visited)
                    visited.remove(edge.target_lct)
        
        _dfs(self.edges, target_lct, [], set())
        return paths
    
    def merge_graphs(self, other: 'MRHGraph', 
                    merge_probability: float = 0.8) -> 'MRHGraph':
        """Merge another MRH graph with probability weighting"""
        merged = MRHGraph(f"{self.entity_id}_merged")
        
        # Add edges from self
        for edge in self.edges:
            merged.add_relevance(edge)
        
        # Add edges from other with probability adjustment
        for edge in other.edges:
            adjusted_edge = MRHEdge(
                target_lct=edge.target_lct,
                probability=edge.probability * merge_probability,
                relation=edge.relation,
                distance=edge.distance + 1,
                decay_rate=edge.decay_rate,
                conditional_on=edge.conditional_on,
                metadata={**edge.metadata, "merged_from": other.entity_id}
            )
            merged.add_relevance(adjusted_edge)
        
        return merged

class LCT:
    """Linked Context Token with RDF-based MRH"""
    
    def __init__(self, entity_id: str, content: Dict):
        self.entity_id = entity_id
        self.content = content
        self.mrh = MRHGraph(entity_id)
        self.timestamp = time.time()
        self.hash = self._compute_hash()
        
    def _compute_hash(self) -> str:
        """Compute LCT hash"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def add_relevance(self, target_lct: 'LCT', 
                     probability: float,
                     relation: MRHRelation,
                     **kwargs) -> None:
        """Add a relevance relationship to another LCT"""
        edge = MRHEdge(
            target_lct=target_lct.hash,
            probability=probability,
            relation=relation,
            **kwargs
        )
        self.mrh.add_relevance(edge)
    
    def to_json(self) -> Dict:
        """Serialize LCT with RDF MRH"""
        return {
            "lct_version": "1.1",
            "entity_id": self.entity_id,
            "hash": self.hash,
            "timestamp": self.timestamp,
            "content": self.content,
            "mrh": self.mrh.to_jsonld()
        }
    
    def find_relevant_contexts(self, min_probability: float = 0.3) -> List[Tuple[str, float]]:
        """Find all relevant contexts above probability threshold"""
        traversal = self.mrh.traverse_markov(min_probability=min_probability)
        return [(lct_id, prob) for lct_id, prob, _ in traversal]

class FractalNavigator:
    """Navigate the fractal graph of graphs"""
    
    def __init__(self):
        self.lct_cache: Dict[str, LCT] = {}
        self.traversal_history: List[str] = []
        
    def register_lct(self, lct: LCT) -> None:
        """Register an LCT in the cache"""
        self.lct_cache[lct.hash] = lct
        
    def traverse_fractal(self, start_lct: LCT, 
                         depth: int = 3,
                         breadth: int = 5) -> nx.DiGraph:
        """
        Traverse the fractal MRH structure, building a NetworkX graph
        """
        G = nx.DiGraph()
        visited = set()
        
        def _traverse(lct: LCT, current_depth: int, parent_prob: float = 1.0):
            if current_depth >= depth or lct.hash in visited:
                return
            
            visited.add(lct.hash)
            self.traversal_history.append(lct.hash)
            
            # Add node
            G.add_node(lct.hash, 
                      entity_id=lct.entity_id,
                      depth=current_depth,
                      probability=parent_prob)
            
            # Get top relevances
            relevances = lct.mrh.traverse_markov(max_depth=1, min_probability=0.1)
            
            for i, (target_hash, prob, _) in enumerate(relevances[:breadth]):
                # Add edge
                G.add_edge(lct.hash, target_hash, 
                          weight=prob,
                          relation="relevance")
                
                # Recursively traverse if LCT is in cache
                if target_hash in self.lct_cache:
                    target_lct = self.lct_cache[target_hash]
                    _traverse(target_lct, current_depth + 1, parent_prob * prob)
        
        _traverse(start_lct, 0)
        return G
    
    def visualize_traversal(self, G: nx.DiGraph, filename: str = "mrh_fractal.png"):
        """Visualize the fractal MRH structure"""
        plt.figure(figsize=(12, 8))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color by depth
        depths = [G.nodes[node].get('depth', 0) for node in G.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, 
                             node_color=depths,
                             cmap='viridis',
                             node_size=500,
                             alpha=0.8)
        
        nx.draw_networkx_labels(G, pos,
                               labels={n: n[:8] for n in G.nodes()},
                               font_size=8)
        
        # Draw edges with probability as width
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos,
                              width=[w * 3 for w in weights],
                              alpha=0.5,
                              edge_color='gray')
        
        plt.title("Fractal MRH Graph Traversal")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        
        print(f"Visualization saved to {filename}")

def demo_mrh_rdf():
    """Demonstrate MRH RDF implementation"""
    
    print("=" * 60)
    print("MRH RDF Implementation Demo")
    print("=" * 60)
    
    # Create LCTs representing a reasoning chain
    premise1 = LCT("entity:premise1", {
        "type": "observation",
        "content": "Temperature is rising"
    })
    
    premise2 = LCT("entity:premise2", {
        "type": "observation", 
        "content": "Ice is melting"
    })
    
    hypothesis = LCT("entity:hypothesis", {
        "type": "inference",
        "content": "Climate change is accelerating"
    })
    
    conclusion = LCT("entity:conclusion", {
        "type": "conclusion",
        "content": "Immediate action required"
    })
    
    alternative = LCT("entity:alternative", {
        "type": "alternative",
        "content": "Natural cycle variation"
    })
    
    # Build MRH relationships
    premise1.add_relevance(
        premise2, 
        probability=0.85,
        relation=MRHRelation.REFERENCES,
        distance=1
    )
    
    premise2.add_relevance(
        hypothesis,
        probability=0.75,
        relation=MRHRelation.PRODUCES,
        distance=1
    )
    
    hypothesis.add_relevance(
        conclusion,
        probability=0.9,
        relation=MRHRelation.PRODUCES,
        distance=1
    )
    
    hypothesis.add_relevance(
        alternative,
        probability=0.1,
        relation=MRHRelation.ALTERNATIVES_TO,
        distance=1
    )
    
    # Add circular reference for fractal structure
    conclusion.add_relevance(
        premise1,
        probability=0.3,
        relation=MRHRelation.DEPENDS_ON,
        distance=3
    )
    
    # Print JSON-LD representation
    print("\n1. LCT with RDF MRH (JSON-LD format):")
    print("-" * 40)
    hypothesis_json = hypothesis.to_json()
    print(json.dumps(hypothesis_json, indent=2))
    
    # Traverse Markovian relevance
    print("\n2. Markovian Traversal from Hypothesis:")
    print("-" * 40)
    relevant = hypothesis.find_relevant_contexts(min_probability=0.05)
    for lct_id, prob in relevant:
        print(f"  {lct_id}: {prob:.3f}")
    
    # Demonstrate fractal navigation
    print("\n3. Fractal Graph Navigation:")
    print("-" * 40)
    
    navigator = FractalNavigator()
    navigator.register_lct(premise1)
    navigator.register_lct(premise2)
    navigator.register_lct(hypothesis)
    navigator.register_lct(conclusion)
    navigator.register_lct(alternative)
    
    # Traverse fractal structure
    G = navigator.traverse_fractal(hypothesis, depth=3, breadth=3)
    
    print(f"  Nodes discovered: {len(G.nodes())}")
    print(f"  Edges created: {len(G.edges())}")
    print(f"  Traversal order: {' -> '.join(navigator.traversal_history[:5])}")
    
    # Visualize (if matplotlib available)
    try:
        navigator.visualize_traversal(G, "mrh_fractal_demo.png")
    except:
        print("  (Visualization skipped - matplotlib not available)")
    
    # Demonstrate graph merging
    print("\n4. Graph Merging (Consensus Building):")
    print("-" * 40)
    
    # Create alternative perspective
    alt_perspective = LCT("entity:alt_perspective", {
        "type": "alternative_view",
        "content": "Different interpretation"
    })
    
    alt_perspective.add_relevance(
        premise1,
        probability=0.6,
        relation=MRHRelation.CONTRADICTS,
        distance=1
    )
    
    # Merge graphs
    merged = hypothesis.mrh.merge_graphs(alt_perspective.mrh, merge_probability=0.5)
    print(f"  Original edges: {len(hypothesis.mrh.edges)}")
    print(f"  Alternative edges: {len(alt_perspective.mrh.edges)}")
    print(f"  Merged edges: {len(merged.edges)}")
    
    # Show SPARQL-like query capability
    print("\n5. SPARQL-like Query on RDF Graph:")
    print("-" * 40)
    
    query = """
    SELECT ?target ?prob WHERE {
        ?relevance mrh:target ?target .
        ?relevance mrh:probability ?prob .
        ?relevance mrh:relation mrh:produces .
        FILTER(?prob > 0.5)
    }
    """
    
    print(f"  Query: Find all 'produces' relations with probability > 0.5")
    
    # Execute query (simplified)
    results = []
    for edge in hypothesis.mrh.edges:
        if edge.relation == MRHRelation.PRODUCES and edge.probability > 0.5:
            results.append((edge.target_lct, edge.probability))
    
    for target, prob in results:
        print(f"    -> {target}: {prob:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)

if __name__ == "__main__":
    demo_mrh_rdf()