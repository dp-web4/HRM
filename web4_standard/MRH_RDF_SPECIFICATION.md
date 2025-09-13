# Web4 Standard Extension: MRH as RDF Graph

## Version 1.0 - Markov Relevancy Horizon RDF Specification

### Author: Dennis Palatov
### Specification Version: 1.0
### Date: 2025

### Abstract

This specification extends the Web4 standard to define the Markov Relevancy Horizon (MRH) field within Linked Context Tokens (LCTs) as an RDF graph rather than a simple list. The MRH concept, created by Dennis Palatov, extends the information-theoretic Markov blanket to explicitly encompass fractal scales, enabling systems to be considered not only in informational context but in relevant fractal context. This specification details how MRH enables fractal graph composition, contextual precision, and semantic traversal while maintaining local flexibility.

## 1. Theoretical Foundation

The Markov Relevancy Horizon extends the classical concept of Markov blankets from probability theory. While a Markov blanket defines the set of nodes that makes a node conditionally independent from the rest of a network, MRH introduces:

- **Relevance over Independence**: Focus on contextual relevance rather than statistical independence
- **Fractal Structure**: Explicit support for self-similar patterns at multiple scales
- **Probabilistic Horizons**: Gradient boundaries rather than hard cutoffs
- **Semantic Relationships**: Typed edges that preserve meaning

This innovation enables systems to maintain context across fractal scales of organization, from individual thoughts to entire organizations.

## 2. Motivation for RDF Representation

The current MRH specification as a flat list of LCT references lacks:
- **Semantic relationships** between referenced contexts
- **Probabilistic weights** for Markovian transitions
- **Contextual metadata** about why references are relevant
- **Fractal composability** for multi-scale navigation

By adopting RDF graphs, MRH becomes a rich semantic structure that naturally forms graphs of graphs, enabling precise context preservation and intelligent traversal.

## 3. Core Specification

### 3.1 Structure

The MRH field in an LCT SHALL be either:
1. **Simple form** (backward compatible): Array of LCT URIs
2. **Graph form**: RDF graph in JSON-LD format

```json
{
  "lct_version": "1.0",
  "entity_id": "entity:uuid",
  "mrh": {
    "@context": {
      "@vocab": "https://web4.foundation/mrh/v1#",
      "mrh": "https://web4.foundation/mrh/v1#",
      "lct": "https://web4.foundation/lct/",
      "xsd": "http://www.w3.org/2001/XMLSchema#"
    },
    "@graph": [
      {
        "@id": "_:relevance1",
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:hash1"},
        "mrh:probability": {"@value": "0.95", "@type": "xsd:decimal"},
        "mrh:relation": "mrh:derives_from",
        "mrh:distance": {"@value": "1", "@type": "xsd:integer"}
      },
      {
        "@id": "_:relevance2",
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:hash2"},
        "mrh:probability": {"@value": "0.7", "@type": "xsd:decimal"},
        "mrh:relation": "mrh:references",
        "mrh:distance": {"@value": "2", "@type": "xsd:integer"},
        "mrh:conditional_on": {"@id": "_:relevance1"}
      }
    ]
  }
}
```

### 3.2 Core Ontology

#### Classes

- `mrh:Relevance` - A relevancy relationship to another LCT
- `mrh:Transition` - A Markovian state transition
- `mrh:Cluster` - A group of related LCTs
- `mrh:Path` - A traversal path through the graph

#### Properties

##### Probability Properties
- `mrh:probability` - Markovian transition probability [0,1]
- `mrh:joint_probability` - Joint probability with dependencies
- `mrh:conditional_on` - Conditional dependency on another relevance

##### Relationship Properties
- `mrh:derives_from` - This LCT derives from target
- `mrh:specializes` - This LCT specializes target concept
- `mrh:contradicts` - This LCT contradicts target
- `mrh:extends` - This LCT extends target
- `mrh:references` - Generic reference
- `mrh:depends_on` - Functional dependency
- `mrh:alternatives_to` - Mutually exclusive alternatives

##### Structural Properties
- `mrh:distance` - Markov distance (integer hops)
- `mrh:trust` - Trust weight for this edge [0,1]
- `mrh:timestamp` - When this relevance was established
- `mrh:decay_rate` - How quickly relevance decays

### 3.3 Fractal Composition

When an LCT's MRH references another LCT, that target LCT contains its own MRH graph. This creates a fractal structure:

```
LCT_A
  └── MRH Graph
      ├── Reference to LCT_B (p=0.9)
      │   └── MRH Graph
      │       ├── Reference to LCT_C (p=0.8)
      │       └── Reference to LCT_D (p=0.6)
      └── Reference to LCT_E (p=0.7)
          └── MRH Graph
              └── Reference to LCT_C (p=0.5)
```

### 3.4 Traversal Semantics

#### Depth-First Markovian Walk
```sparql
SELECT ?lct ?total_prob WHERE {
  ?relevance mrh:target ?lct .
  ?relevance mrh:probability ?prob .
  ?relevance mrh:distance ?dist .
  FILTER(?dist <= 3)
  BIND(?prob * power(0.9, ?dist) as ?total_prob)
}
ORDER BY DESC(?total_prob)
```

#### Breadth-First Relevance Expansion
```javascript
function expandMRH(lct, depth = 3) {
  const graph = [];
  const queue = [{lct, prob: 1.0, dist: 0}];
  
  while (queue.length > 0) {
    const {lct, prob, dist} = queue.shift();
    if (dist >= depth) continue;
    
    const mrh = await fetchLCT(lct).mrh;
    for (const relevance of mrh['@graph']) {
      const newProb = prob * relevance['mrh:probability'];
      if (newProb > threshold) {
        graph.push(relevance);
        queue.push({
          lct: relevance['mrh:target']['@id'],
          prob: newProb,
          dist: dist + 1
        });
      }
    }
  }
  return graph;
}
```

## 4. Use Cases

### 4.1 Context Preservation
When an agent processes information, it maintains the full relevancy graph:
- Input contexts with probabilities
- Transformation relationships
- Output contexts with derivation paths

### 4.2 Trust Propagation
Trust flows through the MRH graph:
```
trust(B) = Σ(trust(A) * P(A→B) * decay(distance))
```

### 4.3 Semantic Search
Find all LCTs within semantic distance:
```sparql
CONSTRUCT {
  ?lct mrh:relevant_to ?query .
  ?lct mrh:path ?path .
  ?lct mrh:combined_probability ?prob .
}
WHERE {
  ?start mrh:contains ?query .
  ?path mrh:from ?start .
  ?path mrh:to ?lct .
  ?path mrh:total_probability ?prob .
  FILTER(?prob > 0.3)
}
```

## 5. Implementation Guidelines

### 5.1 Storage
- Store MRH graphs in triple stores for efficient querying
- Cache expanded graphs for frequently accessed LCTs
- Use named graphs to maintain provenance

### 5.2 Optimization
- Prune low-probability edges (below threshold)
- Implement decay functions for temporal relevance
- Use bloom filters for quick existence checks

### 5.3 Compatibility
- Support both simple array and RDF graph formats
- Provide migration tools from v1.0 to v1.1
- Implement fallback for clients that don't understand RDF

## 6. Examples

### 6.1 Simple Chain of Reasoning
```json
{
  "mrh": {
    "@context": "https://web4.foundation/mrh/v1",
    "@graph": [
      {
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:premise1"},
        "mrh:probability": 0.9,
        "mrh:relation": "mrh:derives_from"
      },
      {
        "@type": "mrh:Relevance", 
        "mrh:target": {"@id": "lct:premise2"},
        "mrh:probability": 0.85,
        "mrh:relation": "mrh:derives_from"
      },
      {
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:conclusion"},
        "mrh:probability": 0.76,
        "mrh:relation": "mrh:produces"
      }
    ]
  }
}
```

### 6.2 Alternative Paths
```json
{
  "mrh": {
    "@context": "https://web4.foundation/mrh/v1",
    "@graph": [
      {
        "@id": "_:path1",
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:solution_a"},
        "mrh:probability": 0.6,
        "mrh:relation": "mrh:alternatives_to",
        "mrh:alternatives": {"@id": "_:path2"}
      },
      {
        "@id": "_:path2",
        "@type": "mrh:Relevance",
        "mrh:target": {"@id": "lct:solution_b"},
        "mrh:probability": 0.4,
        "mrh:relation": "mrh:alternatives_to",
        "mrh:alternatives": {"@id": "_:path1"}
      }
    ]
  }
}
```

### 6.3 Conditional Dependencies
```json
{
  "mrh": {
    "@context": "https://web4.foundation/mrh/v1",
    "@graph": [
      {
        "@id": "_:base",
        "mrh:target": {"@id": "lct:foundation"},
        "mrh:probability": 1.0
      },
      {
        "@id": "_:cond1",
        "mrh:target": {"@id": "lct:extension1"},
        "mrh:probability": 0.8,
        "mrh:conditional_on": {"@id": "_:base"}
      },
      {
        "@id": "_:cond2",
        "mrh:target": {"@id": "lct:extension2"},
        "mrh:probability": 0.7,
        "mrh:conditional_on": [
          {"@id": "_:base"},
          {"@id": "_:cond1"}
        ],
        "mrh:joint_probability": 0.56
      }
    ]
  }
}
```

## 7. Benefits

### 7.1 Contextual Precision
- Every link carries its semantic relationship
- Probabilities enable intelligent traversal
- Conditions and dependencies are explicit

### 7.2 Local Flexibility
- Each LCT can extend the ontology
- Communities develop specialized predicates
- Backward compatibility maintained

### 7.3 Fractal Scalability
- Navigate at any level of detail
- Compose graphs without losing context
- Natural sharding boundaries

### 7.4 Markovian Properties
- True Markov property: future depends only on current state
- Probability propagation through the graph
- Optimal stopping conditions for traversal

## 8. Migration Path

### Phase 1: Dual Support
- Accept both array and graph formats
- Convert arrays to simple graphs internally
- Provide tools for upgrade

### Phase 2: Adoption
- Encourage graph format for new LCTs
- Provide libraries and tools
- Build example applications

### Phase 3: Deprecation
- Mark array format as legacy
- Full ecosystem on graph format
- Maintain compatibility layer

## 9. Conclusion

By adopting RDF graphs for MRH, the Web4 standard gains:
- **Semantic precision** through typed relationships
- **Probabilistic reasoning** through Markovian properties
- **Fractal composition** through graphs of graphs
- **Contextual coherence** through preserved relationships

This positions Web4 as a truly semantic, probabilistic, and scalable framework for the next generation of distributed intelligence.

## Attribution

The Markov Relevancy Horizon concept was created by **Dennis Palatov** to extend the information-theoretic concept of Markov blankets to explicitly encompass fractal scales, enabling a new class of context-aware distributed systems.

---

*"The Markov Relevancy Horizon is not just about what is relevant, but how and why it is relevant, with what probability, and under what conditions."* - Dennis Palatov