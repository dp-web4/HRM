# Web4 Standard & SAGE Architecture Glossary

## Core Acronyms & Terms

### Web4 Foundation

#### LCT - Linked Context Token
- **Definition**: The fundamental unit of the Web4 standard that encapsulates context with cryptographic proofs
- **Components**: Entity ID, MRH, signatures, metadata
- **Purpose**: Enables verifiable context preservation across distributed systems

#### MRH - Markov Relevancy Horizon  
- **Definition**: The probabilistic graph of relevant contexts within an LCT
- **Format**: RDF graph (v1.1) or simple array (v1.0)
- **Purpose**: Captures relevance relationships with probabilities and semantic predicates
- **Origin**: Created by Dennis Palatov to expand the concept of Markov blanket to explicitly encompass fractal scales, enabling systems to be considered not only in informational context but in relevant fractal context
- **Theoretical Foundation**: Extends the established information-theoretic concept of Markov blanket to support fractal composition
- **Note**: NOT "Markovian" - always use "Markov"

#### RDF - Resource Description Framework
- **Definition**: W3C standard for data interchange using subject-predicate-object triples
- **Usage**: MRH graphs are expressed as RDF in JSON-LD format
- **Purpose**: Enables semantic relationships and SPARQL queries

### SAGE Architecture

#### SAGE - Strategic Augmentation & Generative Enhancement
- **Definition**: Hierarchical reasoning architecture for edge AI systems
- **Components**: H-level (strategic) and L-level (tactical) modules
- **Purpose**: Enable efficient reasoning on resource-constrained devices

#### HRM - Hierarchical Reasoning Module
- **Definition**: The core implementation framework for SAGE
- **Size**: 27M parameters
- **Purpose**: Demonstrates hierarchical separation of reasoning concerns

#### H-Level - High-level/Strategic Layer
- **Definition**: The strategic reasoning layer in SAGE/HRM
- **Functions**: Pattern recognition, planning, routing decisions
- **Analogy**: "System 2" thinking, conscious deliberation

#### L-Level - Low-level/Tactical Layer
- **Definition**: The tactical execution layer in SAGE/HRM
- **Functions**: Precise operations, skill execution, verification
- **Analogy**: "System 1" thinking, automatic responses

#### IRP - Intelligent Routing Protocol
- **Definition**: Plugin architecture for L-level specialist modules
- **Purpose**: Dynamic selection of specialized attention heads
- **Note**: Maps directly to L-level specialists in cascading architecture

### Technical Components

#### PBM - Peripheral Broadcast Mailbox
- **Definition**: GPU mailbox for many-to-many fixed-size records
- **Purpose**: Efficient inter-tile communication in GPU architectures

#### FTM - Focus Tensor Mailbox
- **Definition**: GPU mailbox for zero-copy tensor pointer handoff
- **Purpose**: Large tensor sharing between GPU tiles

#### SPARQL - SPARQL Protocol and RDF Query Language
- **Definition**: Query language for RDF graphs
- **Usage**: Query MRH graphs for patterns, paths, and relationships
- **Note**: Recursive acronym (SPARQL = SPARQL Protocol...)

#### JSON-LD - JSON for Linking Data
- **Definition**: JSON-based format for RDF data
- **Usage**: MRH graphs are serialized as JSON-LD
- **Purpose**: Human-readable RDF with familiar JSON syntax

### Reasoning Concepts

#### ARC - Abstraction and Reasoning Corpus
- **Definition**: Benchmark for abstract reasoning capabilities
- **Relevance**: Demonstrates need for hierarchical reasoning
- **Note**: Not "ARC-AGI" in our context, just "ARC"

#### VAE - Variational Autoencoder
- **Definition**: Neural network for learning compressed representations
- **Usage**: TinyVAE for edge-optimized compression
- **Purpose**: Enable efficient latent space operations

#### KV-Cache - Key-Value Cache
- **Definition**: Attention state storage in transformers
- **Purpose**: Enables consciousness persistence and resumption
- **Size**: Critical factor for generation throughput

### Graph Relationships (MRH Predicates)

#### mrh:derives_from
- **Definition**: This LCT derives from the target LCT
- **Usage**: Source documents, foundational contexts
- **Direction**: Current → Past

#### mrh:extends
- **Definition**: This LCT extends the target LCT
- **Usage**: Future work, elaborations
- **Direction**: Current → Future

#### mrh:references
- **Definition**: Generic reference relationship
- **Usage**: Citations, related work
- **Direction**: Bidirectional

#### mrh:contradicts
- **Definition**: This LCT contradicts the target LCT
- **Usage**: Conflicting information, corrections
- **Direction**: Bidirectional opposition

#### mrh:alternatives_to
- **Definition**: Mutually exclusive alternative approaches
- **Usage**: Different solutions to same problem
- **Direction**: Symmetric relationship

#### mrh:depends_on
- **Definition**: Functional dependency on target LCT
- **Usage**: Prerequisites, requirements
- **Direction**: Current requires Target

#### mrh:produces
- **Definition**: This LCT produces the target LCT
- **Usage**: Outputs, results, conclusions
- **Direction**: Current → Result

#### mrh:specializes
- **Definition**: This LCT is a specialization of target
- **Usage**: Specific instances of general concepts
- **Direction**: Specific → General

### Trust & Probability Terms

#### Trust Propagation
- **Definition**: Flow of trust through MRH graph edges
- **Models**: Multiplicative, Probabilistic, Maximal, Minimal
- **Formula**: trust = source_trust × probability × edge_trust × decay

#### Markov Property
- **Definition**: Future state depends only on current state, not history
- **Application**: MRH transitions are Markovian
- **Note**: Always "Markov" not "Markovian" for the horizon itself

#### Decay Factor
- **Definition**: Rate at which relevance/trust decreases with distance
- **Default**: 0.9 per hop in MRH graphs
- **Purpose**: Prioritize local context over distant

### Implementation Terms

#### Fractal Composition
- **Definition**: Self-similar structure at multiple scales
- **Application**: MRH graphs contain MRH graphs
- **Purpose**: Natural hierarchical organization

#### Theatrical Reasoning
- **Definition**: Observable routing decisions in LLMs
- **Reality**: Actual architectural patterns, not performance
- **Validation**: Confirmed by Jet-Nemotron results

#### Cascading Architecture
- **Definition**: H-level routes to L-level specialists
- **Discovered**: Pattern found in both SAGE and Jet-Nemotron
- **Efficiency**: 47× speedup with selective attention

### Platform-Specific Terms

#### Jetson (Orin Nano)
- **Definition**: NVIDIA edge AI platform
- **Specs**: 8GB unified memory, 1024 CUDA cores
- **Role**: Primary deployment target for SAGE

#### Legion (Pro 7)
- **Definition**: Development machine with RTX 4090
- **Purpose**: High-performance testing and training
- **Note**: Not "Legion Pro", just "Legion"

#### WSL2 - Windows Subsystem for Linux 2
- **Definition**: Linux environment on Windows
- **Usage**: Development environment
- **Note**: Always "WSL2" not "WSL 2" or "WSL"

## Usage Guidelines

### Consistency Rules

1. **Acronyms**: Always use the exact acronym as defined (e.g., MRH not Mrh or mrh in text)
2. **First Use**: Spell out acronym on first use in documents
3. **Capitalization**: 
   - Acronyms: All caps (MRH, LCT, SAGE)
   - Concepts: Title case (Markov Relevancy Horizon)
   - Code: Follow language conventions
4. **Predicates**: Always use namespace prefix (mrh:derives_from, not just derives_from)
5. **No Variations**: 
   - MRH = Markov Relevancy Horizon (NOT Markovian)
   - H-level (NOT H-layer or High-level when referring to architecture)
   - L-level (NOT L-layer or Low-level when referring to architecture)

### Common Mistakes to Avoid

- ❌ "Markovian Relevancy Horizon" → ✅ "Markov Relevancy Horizon"
- ❌ "HRM module" → ✅ "HRM" (already means Hierarchical Reasoning Module)
- ❌ "RDF-graph" → ✅ "RDF graph" (no hyphen)
- ❌ "Trust-propagation" → ✅ "Trust propagation" (no hyphen)
- ❌ "H-layer/L-layer" → ✅ "H-level/L-level" (levels not layers)
- ❌ "SAGE framework" → ✅ "SAGE architecture" (it's an architecture)
- ❌ "Web 4" or "Web-4" → ✅ "Web4" (one word, no space/hyphen)

## Version History

- **v1.0** (2025-01-13): Initial glossary creation
- Terms will be updated as the Web4 standard evolves