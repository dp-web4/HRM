"""
MRH Graph Visualization Tool
============================

Creates interactive visualizations of Markov Relevancy Horizon graphs
for understanding the fractal structure of Web4 LCTs.
"""

import json
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphNode:
    """Node in the MRH visualization"""
    id: str
    label: str
    probability: float
    distance: int
    x: float = 0.0
    y: float = 0.0
    size: float = 1.0
    color: str = "#4CAF50"


@dataclass
class GraphEdge:
    """Edge in the MRH visualization"""
    source: str
    target: str
    relation: str
    probability: float
    conditional: Optional[str] = None
    color: str = "#999999"
    width: float = 1.0


class MRHVisualizer:
    """Visualizes MRH graphs as interactive HTML/SVG"""
    
    # Relation type colors
    RELATION_COLORS = {
        "derives_from": "#2196F3",      # Blue - derivation
        "references": "#4CAF50",        # Green - reference
        "extends": "#9C27B0",          # Purple - extension
        "contradicts": "#F44336",      # Red - contradiction
        "alternatives_to": "#FF9800",   # Orange - alternatives
        "depends_on": "#00BCD4",       # Cyan - dependency
        "specializes": "#3F51B5",      # Indigo - specialization
    }
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.current_lct = "current"
    
    def parse_mrh(self, lct_data: Dict[str, Any]) -> None:
        """Parse an LCT's MRH data into graph structure"""
        self.nodes.clear()
        self.edges.clear()
        
        # Add current LCT as center node
        self.nodes[self.current_lct] = GraphNode(
            id=self.current_lct,
            label="Current LCT",
            probability=1.0,
            distance=0,
            size=2.0,
            color="#FF5722"
        )
        
        # Parse MRH graph
        if "mrh" in lct_data and "@graph" in lct_data["mrh"]:
            for relevance in lct_data["mrh"]["@graph"]:
                self._parse_relevance(relevance)
    
    def _parse_relevance(self, relevance: Dict[str, Any]) -> None:
        """Parse a single relevance entry"""
        # Extract target
        target_data = relevance.get("mrh:target", {})
        if isinstance(target_data, dict):
            target_id = target_data.get("@id", "unknown")
        else:
            target_id = str(target_data)
        
        # Clean target ID
        target_label = target_id.split(":")[-1] if ":" in target_id else target_id.split("/")[-1]
        
        # Extract properties
        prob = float(relevance.get("mrh:probability", {}).get("@value", 0.5))
        dist = int(relevance.get("mrh:distance", {}).get("@value", 1))
        relation = relevance.get("mrh:relation", "references")
        
        # Clean relation name
        if "#" in relation:
            relation = relation.split("#")[-1]
        elif "/" in relation:
            relation = relation.split("/")[-1]
        elif ":" in relation:
            relation = relation.split(":")[-1]
        
        # Add node if not exists
        if target_id not in self.nodes:
            self.nodes[target_id] = GraphNode(
                id=target_id,
                label=target_label,
                probability=prob,
                distance=dist,
                size=1.5 * prob,  # Size based on probability
                color=self._get_node_color(prob)
            )
        
        # Add edge
        edge_color = self.RELATION_COLORS.get(relation, "#999999")
        self.edges.append(GraphEdge(
            source=self.current_lct,
            target=target_id,
            relation=relation,
            probability=prob,
            color=edge_color,
            width=3.0 * prob  # Width based on probability
        ))
        
        # Handle conditional dependencies
        if "mrh:conditional_on" in relevance:
            condition = relevance["mrh:conditional_on"]
            if isinstance(condition, dict):
                condition_id = condition.get("@id", "")
            else:
                condition_id = str(condition)
            
            if condition_id:
                self.edges.append(GraphEdge(
                    source=condition_id,
                    target=target_id,
                    relation="condition",
                    probability=0.5,
                    color="#FFC107",
                    width=1.0
                ))
    
    def _get_node_color(self, probability: float) -> str:
        """Get node color based on probability"""
        # Gradient from red (low) to green (high)
        if probability < 0.33:
            return "#F44336"  # Red
        elif probability < 0.67:
            return "#FF9800"  # Orange
        else:
            return "#4CAF50"  # Green
    
    def layout_graph(self) -> None:
        """Apply force-directed layout to graph"""
        # Simple circular layout by distance
        for node_id, node in self.nodes.items():
            if node_id == self.current_lct:
                node.x = 400
                node.y = 300
            else:
                # Place nodes in circles by distance
                angle = hash(node_id) % 360
                radius = 100 + (node.distance * 80)
                node.x = 400 + radius * math.cos(math.radians(angle))
                node.y = 300 + radius * math.sin(math.radians(angle))
    
    def generate_svg(self, width: int = 800, height: int = 600) -> str:
        """Generate SVG visualization"""
        self.layout_graph()
        
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<style>',
            '.node { cursor: pointer; }',
            '.node:hover { opacity: 0.8; }',
            '.edge { stroke-linecap: round; opacity: 0.6; }',
            '.label { font-family: Arial, sans-serif; font-size: 12px; pointer-events: none; }',
            '</style>',
            '<defs>',
            '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '<polygon points="0 0, 10 3.5, 0 7" fill="#666" />',
            '</marker>',
            '</defs>'
        ]
        
        # Draw edges
        svg_parts.append('<g id="edges">')
        for edge in self.edges:
            if edge.source in self.nodes and edge.target in self.nodes:
                source = self.nodes[edge.source]
                target = self.nodes[edge.target]
                
                svg_parts.append(
                    f'<line class="edge" x1="{source.x}" y1="{source.y}" '
                    f'x2="{target.x}" y2="{target.y}" '
                    f'stroke="{edge.color}" stroke-width="{edge.width}" '
                    f'marker-end="url(#arrowhead)" opacity="{edge.probability}">'
                    f'<title>{edge.relation} (p={edge.probability:.2f})</title>'
                    f'</line>'
                )
        svg_parts.append('</g>')
        
        # Draw nodes
        svg_parts.append('<g id="nodes">')
        for node in self.nodes.values():
            svg_parts.append(
                f'<circle class="node" cx="{node.x}" cy="{node.y}" '
                f'r="{10 * node.size}" fill="{node.color}" stroke="#333" stroke-width="2">'
                f'<title>{node.label} (p={node.probability:.2f}, d={node.distance})</title>'
                f'</circle>'
            )
            
            # Add labels
            svg_parts.append(
                f'<text class="label" x="{node.x}" y="{node.y + 25}" '
                f'text-anchor="middle" fill="#333">{node.label}</text>'
            )
        svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def generate_html(self, title: str = "MRH Graph Visualization") -> str:
        """Generate complete HTML page with interactive visualization"""
        svg = self.generate_svg()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }}
        .visualization {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
            background: #f5f5f5;
            border-radius: 5px;
            padding: 20px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Markov Relevancy Horizon - Fractal Graph Structure</p>
        
        <div class="visualization">
            {svg}
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2196F3;"></div>
                <span>Derives From</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #4CAF50;"></div>
                <span>References</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9C27B0;"></div>
                <span>Extends</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F44336;"></div>
                <span>Contradicts</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF9800;"></div>
                <span>Alternatives</span>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(self.nodes)}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self.edges)}</div>
                <div class="stat-label">Total Edges</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max([n.distance for n in self.nodes.values()])}</div>
                <div class="stat-label">Max Distance</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum([e.probability for e in self.edges])/len(self.edges):.2f}</div>
                <div class="stat-label">Avg Probability</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        return html
    
    def save_visualization(self, output_path: str, lct_data: Dict[str, Any]) -> None:
        """Save visualization to HTML file"""
        self.parse_mrh(lct_data)
        html = self.generate_html()
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Visualization saved to {output_path}")


def demonstrate_visualizer():
    """Demonstrate the MRH visualizer"""
    
    # Example LCT with complex MRH
    example_lct = {
        "lct_version": "1.0",
        "entity_id": "entity:research_system",
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
                    "mrh:target": {"@id": "lct:foundation"},
                    "mrh:probability": {"@value": "0.95", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:derives_from",
                    "mrh:distance": {"@value": "1", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:theory_1"},
                    "mrh:probability": {"@value": "0.85", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:references",
                    "mrh:distance": {"@value": "2", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:theory_2"},
                    "mrh:probability": {"@value": "0.75", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:references",
                    "mrh:distance": {"@value": "2", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:extension"},
                    "mrh:probability": {"@value": "0.8", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:extends",
                    "mrh:distance": {"@value": "1", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:alternative_1"},
                    "mrh:probability": {"@value": "0.6", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:alternatives_to",
                    "mrh:distance": {"@value": "2", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:alternative_2"},
                    "mrh:probability": {"@value": "0.4", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:alternatives_to",
                    "mrh:distance": {"@value": "2", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:contradiction"},
                    "mrh:probability": {"@value": "0.3", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:contradicts",
                    "mrh:distance": {"@value": "3", "@type": "xsd:integer"}
                },
                {
                    "@type": "mrh:Relevance",
                    "mrh:target": {"@id": "lct:specialization"},
                    "mrh:probability": {"@value": "0.9", "@type": "xsd:decimal"},
                    "mrh:relation": "mrh:specializes",
                    "mrh:distance": {"@value": "1", "@type": "xsd:integer"}
                }
            ]
        }
    }
    
    # Create visualizer
    viz = MRHVisualizer()
    
    # Save visualization
    viz.save_visualization("mrh_graph_visualization.html", example_lct)
    
    # Also generate just the SVG
    viz.parse_mrh(example_lct)
    svg = viz.generate_svg()
    
    with open("mrh_graph.svg", "w") as f:
        f.write(svg)
    
    print("SVG saved to mrh_graph.svg")
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"  Nodes: {len(viz.nodes)}")
    print(f"  Edges: {len(viz.edges)}")
    print(f"  Relations: {set(e.relation for e in viz.edges)}")
    print(f"  Probability range: {min(e.probability for e in viz.edges):.2f} - {max(e.probability for e in viz.edges):.2f}")


if __name__ == "__main__":
    demonstrate_visualizer()