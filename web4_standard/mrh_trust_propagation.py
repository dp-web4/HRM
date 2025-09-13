"""
Trust Propagation Algorithm for MRH Graphs
==========================================

Implements trust flow through the Markov Relevancy Horizon,
enabling decentralized trust computation across the Web4 network.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum


class TrustModel(Enum):
    """Different trust propagation models"""
    MULTIPLICATIVE = "multiplicative"  # Trust = product of path trusts
    PROBABILISTIC = "probabilistic"    # Trust = 1 - ∏(1 - path_trust)
    WEIGHTED_AVERAGE = "weighted"      # Trust = weighted average
    MAXIMAL = "maximal"               # Trust = max path trust
    MINIMAL = "minimal"               # Trust = min path trust


@dataclass
class TrustPath:
    """Represents a trust path through the graph"""
    nodes: List[str]
    edges: List[Tuple[str, str, float]]  # (source, target, probability)
    total_trust: float
    distance: int
    
    def __str__(self):
        path_str = " → ".join(self.nodes)
        return f"{path_str} (trust: {self.total_trust:.3f})"


@dataclass
class TrustNode:
    """Node with trust metadata"""
    id: str
    base_trust: float = 0.5
    computed_trust: float = 0.0
    incoming_trust: Dict[str, float] = field(default_factory=dict)
    outgoing_trust: Dict[str, float] = field(default_factory=dict)
    visited: bool = False
    iteration_updated: int = -1


class MRHTrustPropagator:
    """
    Trust propagation engine for MRH graphs.
    
    Implements various trust models including:
    - PageRank-style iterative propagation
    - Path-based trust computation
    - Conditional trust with dependencies
    - Temporal decay models
    """
    
    def __init__(self, 
                 model: TrustModel = TrustModel.PROBABILISTIC,
                 decay_factor: float = 0.9,
                 convergence_threshold: float = 0.001,
                 max_iterations: int = 100):
        self.model = model
        self.decay_factor = decay_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
        # Graph structure
        self.nodes: Dict[str, TrustNode] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.conditional_edges: Dict[Tuple[str, str], List[str]] = {}
        
        # Trust computation cache
        self.trust_cache: Dict[str, float] = {}
        self.path_cache: Dict[Tuple[str, str], List[TrustPath]] = {}
    
    def load_mrh_graph(self, lct_data: Dict[str, Any], source_id: str = "current") -> None:
        """Load MRH graph from LCT data"""
        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.conditional_edges.clear()
        self.trust_cache.clear()
        
        # Add source node
        self.nodes[source_id] = TrustNode(id=source_id, base_trust=1.0)
        
        # Parse MRH graph
        if "mrh" in lct_data and "@graph" in lct_data["mrh"]:
            for relevance in lct_data["mrh"]["@graph"]:
                self._parse_relevance(relevance, source_id)
    
    def _parse_relevance(self, relevance: Dict[str, Any], source_id: str) -> None:
        """Parse relevance entry and add to graph"""
        # Extract target
        target_data = relevance.get("mrh:target", {})
        if isinstance(target_data, dict):
            target_id = target_data.get("@id", "unknown")
        else:
            target_id = str(target_data)
        
        # Extract probability and trust
        prob_data = relevance.get("mrh:probability", {})
        if isinstance(prob_data, dict):
            probability = float(prob_data.get("@value", 0.5))
        else:
            probability = float(prob_data) if prob_data else 0.5
        
        trust_data = relevance.get("mrh:trust", {})
        if isinstance(trust_data, dict):
            edge_trust = float(trust_data.get("@value", 1.0))
        else:
            edge_trust = float(trust_data) if trust_data else 1.0
        
        # Combined edge weight
        edge_weight = probability * edge_trust
        
        # Add node if not exists
        if target_id not in self.nodes:
            self.nodes[target_id] = TrustNode(id=target_id, base_trust=0.5)
        
        # Add edge
        self.edges[(source_id, target_id)] = edge_weight
        self.nodes[source_id].outgoing_trust[target_id] = edge_weight
        self.nodes[target_id].incoming_trust[source_id] = edge_weight
        
        # Handle conditional dependencies
        if "mrh:conditional_on" in relevance:
            conditions = relevance["mrh:conditional_on"]
            if not isinstance(conditions, list):
                conditions = [conditions]
            
            condition_ids = []
            for cond in conditions:
                if isinstance(cond, dict):
                    condition_ids.append(cond.get("@id", ""))
                else:
                    condition_ids.append(str(cond))
            
            self.conditional_edges[(source_id, target_id)] = condition_ids
    
    # ============================================
    # Core Trust Propagation Algorithms
    # ============================================
    
    def propagate_trust_iterative(self) -> Dict[str, float]:
        """
        Iterative trust propagation (PageRank-style).
        Trust flows through the network until convergence.
        """
        # Initialize trust scores
        trust_scores = {node_id: node.base_trust for node_id, node in self.nodes.items()}
        
        for iteration in range(self.max_iterations):
            new_scores = {}
            max_change = 0
            
            for node_id, node in self.nodes.items():
                # Base trust (damping factor)
                new_trust = (1 - self.decay_factor) * node.base_trust
                
                # Incoming trust from neighbors
                incoming_trust = 0
                for source_id, weight in node.incoming_trust.items():
                    # Check conditional dependencies
                    if (source_id, node_id) in self.conditional_edges:
                        conditions = self.conditional_edges[(source_id, node_id)]
                        condition_met = all(
                            trust_scores.get(cond, 0) > 0.5 for cond in conditions
                        )
                        if not condition_met:
                            continue
                    
                    # Propagate trust
                    source_trust = trust_scores[source_id]
                    incoming_trust += source_trust * weight
                
                # Apply trust model
                if self.model == TrustModel.MULTIPLICATIVE:
                    new_trust += self.decay_factor * incoming_trust
                elif self.model == TrustModel.PROBABILISTIC:
                    # Combine using probability OR
                    for source_id, weight in node.incoming_trust.items():
                        source_trust = trust_scores[source_id] * weight * self.decay_factor
                        new_trust = new_trust + source_trust - (new_trust * source_trust)
                else:
                    new_trust += self.decay_factor * incoming_trust
                
                # Clamp to [0, 1]
                new_trust = min(1.0, max(0.0, new_trust))
                new_scores[node_id] = new_trust
                
                # Track convergence
                change = abs(new_trust - trust_scores[node_id])
                max_change = max(max_change, change)
            
            # Update scores
            trust_scores = new_scores
            
            # Check convergence
            if max_change < self.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return trust_scores
    
    def compute_path_trust(self, source: str, target: str, max_depth: int = 5) -> List[TrustPath]:
        """
        Compute all trust paths from source to target.
        Returns paths sorted by trust score.
        """
        if (source, target) in self.path_cache:
            return self.path_cache[(source, target)]
        
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str], edges: List[Tuple[str, str, float]], 
                trust: float, depth: int):
            if depth > max_depth:
                return
            
            if current == target:
                paths.append(TrustPath(
                    nodes=path.copy(),
                    edges=edges.copy(),
                    total_trust=trust,
                    distance=len(path) - 1
                ))
                return
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor_id, weight in self.nodes[current].outgoing_trust.items():
                if neighbor_id not in visited:
                    # Compute path trust based on model
                    if self.model == TrustModel.MULTIPLICATIVE:
                        new_trust = trust * weight * (self.decay_factor ** depth)
                    elif self.model == TrustModel.MINIMAL:
                        new_trust = min(trust, weight)
                    elif self.model == TrustModel.MAXIMAL:
                        new_trust = max(trust, weight)
                    else:
                        new_trust = trust * weight * (self.decay_factor ** depth)
                    
                    if new_trust > 0.01:  # Prune low-trust paths
                        path.append(neighbor_id)
                        edges.append((current, neighbor_id, weight))
                        dfs(neighbor_id, path, edges, new_trust, depth + 1)
                        path.pop()
                        edges.pop()
            
            visited.remove(current)
        
        # Start DFS
        dfs(source, [source], [], 1.0, 0)
        
        # Sort by trust
        paths.sort(key=lambda p: p.total_trust, reverse=True)
        
        # Cache results
        self.path_cache[(source, target)] = paths
        
        return paths
    
    def compute_network_trust(self, source: str) -> Dict[str, float]:
        """
        Compute trust scores for entire network from a source node.
        Uses breadth-first propagation with decay.
        """
        trust_scores = defaultdict(float)
        trust_scores[source] = 1.0
        
        # BFS queue: (node_id, current_trust, distance)
        queue = deque([(source, 1.0, 0)])
        processed = set()
        
        while queue:
            current_id, current_trust, distance = queue.popleft()
            
            if current_id in processed:
                continue
            processed.add(current_id)
            
            # Propagate to neighbors
            if current_id in self.nodes:
                for neighbor_id, weight in self.nodes[current_id].outgoing_trust.items():
                    # Apply decay
                    propagated_trust = current_trust * weight * (self.decay_factor ** distance)
                    
                    # Update trust using model
                    if self.model == TrustModel.PROBABILISTIC:
                        # Combine using probability OR
                        old_trust = trust_scores[neighbor_id]
                        new_trust = old_trust + propagated_trust - (old_trust * propagated_trust)
                        trust_scores[neighbor_id] = new_trust
                    elif self.model == TrustModel.MAXIMAL:
                        trust_scores[neighbor_id] = max(trust_scores[neighbor_id], propagated_trust)
                    else:
                        trust_scores[neighbor_id] = max(trust_scores[neighbor_id], propagated_trust)
                    
                    # Continue propagation if trust is significant
                    if propagated_trust > 0.01 and neighbor_id not in processed:
                        queue.append((neighbor_id, propagated_trust, distance + 1))
        
        return dict(trust_scores)
    
    # ============================================
    # Advanced Trust Features
    # ============================================
    
    def compute_conditional_trust(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Compute trust with specific conditions activated.
        Conditions are node_id -> trust_value mappings.
        """
        # Set conditional trust values
        for node_id, trust in conditions.items():
            if node_id in self.nodes:
                self.nodes[node_id].base_trust = trust
        
        # Propagate with conditions
        trust_scores = self.propagate_trust_iterative()
        
        # Reset base trust
        for node_id in conditions:
            if node_id in self.nodes:
                self.nodes[node_id].base_trust = 0.5
        
        return trust_scores
    
    def identify_trust_clusters(self, min_trust: float = 0.7) -> List[Set[str]]:
        """
        Identify clusters of high-trust nodes.
        Nodes in same cluster have mutual trust above threshold.
        """
        clusters = []
        visited = set()
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            # Find cluster using BFS
            cluster = set()
            queue = deque([node_id])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                # Add high-trust neighbors
                for neighbor_id, weight in self.nodes[current].outgoing_trust.items():
                    if weight >= min_trust and neighbor_id not in visited:
                        queue.append(neighbor_id)
                
                for neighbor_id, weight in self.nodes[current].incoming_trust.items():
                    if weight >= min_trust and neighbor_id not in visited:
                        queue.append(neighbor_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def compute_trust_centrality(self) -> Dict[str, float]:
        """
        Compute trust-based centrality scores.
        Nodes with high centrality are trust hubs.
        """
        centrality = {}
        
        for node_id, node in self.nodes.items():
            # Incoming trust strength
            in_trust = sum(node.incoming_trust.values())
            
            # Outgoing trust strength  
            out_trust = sum(node.outgoing_trust.values())
            
            # Number of connections
            degree = len(node.incoming_trust) + len(node.outgoing_trust)
            
            # Weighted centrality
            centrality[node_id] = (in_trust + out_trust) * np.log(degree + 1)
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1.0
        for node_id in centrality:
            centrality[node_id] /= max_centrality
        
        return centrality


def demonstrate_trust_propagation():
    """Demonstrate trust propagation algorithms"""
    
    # Create example LCT with trust values
    example_lct = {
        "lct_version": "1.0",
        "entity_id": "entity:trust_network",
        "mrh": {
            "@context": {
                "@vocab": "https://web4.foundation/mrh/v1#",
                "mrh": "https://web4.foundation/mrh/v1#",
                "lct": "https://web4.foundation/lct/",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            },
            "@graph": [
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:trusted_source"},
                    "mrh:probability": {"@value": "0.95", "@type": "xsd:decimal"},
                    "mrh:trust": {"@value": "0.9", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:derives_from"
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:peer_1"},
                    "mrh:probability": {"@value": "0.8", "@type": "xsd:decimal"},
                    "mrh:trust": {"@value": "0.7", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:references"
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:peer_2"},
                    "mrh:probability": {"@value": "0.75", "@type": "xsd:decimal"},
                    "mrh:trust": {"@value": "0.8", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:references"
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:untrusted"},
                    "mrh:probability": {"@value": "0.9", "@type": "xsd:decimal"},
                    "mrh:trust": {"@value": "0.2", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:contradicts"
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:conditional"},
                    "mrh:probability": {"@value": "0.85", "@type": "xsd:decimal"},
                    "mrh:trust": {"@value": "0.9", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:extends",
                    "mrh:conditional_on": {"@id": "lct:peer_1"}
                }
            ]
        }
    }
    
    print("=" * 60)
    print("MRH Trust Propagation Demonstration")
    print("=" * 60)
    
    # Test different trust models
    models = [
        TrustModel.PROBABILISTIC,
        TrustModel.MULTIPLICATIVE,
        TrustModel.MAXIMAL
    ]
    
    for model in models:
        print(f"\n{model.value.upper()} TRUST MODEL:")
        print("-" * 40)
        
        propagator = MRHTrustPropagator(model=model)
        propagator.load_mrh_graph(example_lct)
        
        # Iterative propagation
        trust_scores = propagator.propagate_trust_iterative()
        print("\nIterative Trust Scores:")
        for node_id, trust in sorted(trust_scores.items(), key=lambda x: x[1], reverse=True):
            node_label = node_id.split(":")[-1] if ":" in node_id else node_id
            print(f"  {node_label}: {trust:.3f}")
        
        # Network trust from source
        network_trust = propagator.compute_network_trust("current")
        print("\nNetwork Trust from Current:")
        for node_id, trust in sorted(network_trust.items(), key=lambda x: x[1], reverse=True):
            node_label = node_id.split(":")[-1] if ":" in node_id else node_id
            print(f"  {node_label}: {trust:.3f}")
    
    # Trust centrality
    print("\n" + "=" * 60)
    print("TRUST CENTRALITY:")
    print("-" * 40)
    
    propagator = MRHTrustPropagator()
    propagator.load_mrh_graph(example_lct)
    centrality = propagator.compute_trust_centrality()
    
    for node_id, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
        node_label = node_id.split(":")[-1] if ":" in node_id else node_id
        print(f"  {node_label}: {score:.3f}")
    
    # Trust clusters
    print("\nTRUST CLUSTERS (min_trust=0.5):")
    print("-" * 40)
    
    clusters = propagator.identify_trust_clusters(min_trust=0.5)
    for i, cluster in enumerate(clusters, 1):
        cluster_labels = [n.split(":")[-1] if ":" in n else n for n in cluster]
        print(f"  Cluster {i}: {', '.join(cluster_labels)}")
    
    print("\n" + "=" * 60)
    print("Trust propagation demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_trust_propagation()