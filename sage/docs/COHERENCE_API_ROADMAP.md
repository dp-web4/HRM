# SAGE Coherence API Roadmap

**Last Updated:** 2025-11-18
**Status:** Design Phase - Core Infrastructure Exists

## Overview

SAGE currently functions as a cognition kernel for edge devices with autonomous operation, curiosity-driven learning, and multi-modal integration. However, it lacks a formal **coherence API** for multi-agent coordination and ecosystem integration.

This document proposes a coherence API that enables SAGE agents to:
- Register themselves in a distributed network
- Request and share context
- Self-update collaboratively
- Report coherence metrics
- Adjust intent dynamically
- Link to semantic resources
- Negotiate compatible context boundaries

## Current SAGE Capabilities

### ✅ Operational Infrastructure

**Core Loop (SAGE Kernel):**
- Continuous inference maintaining state across time
- Trust-based ATP budget allocation
- Metabolic states (WAKE, FOCUS, REST, DREAM, CRISIS)
- SNARC salience (Surprise, Novelty, Arousal, Reward, Conflict)
- Learning what deserves attention

**IRP Framework (Cognition API):**
- Universal plugin interface: `init_state() → step() → energy() → halt()`
- Iterative refinement: noisy → refined until energy stops decreasing
- 15+ working plugins (Vision, Audio, Language, Memory, TTS, Control)
- Trust emerges from convergence behavior

**VAE Translation Layer:**
- Shared latent spaces for cross-modal communication
- TinyVAE: 192× compression (224×224 → 64D latent)
- InformationBottleneck: 16× compression (4096D → 256D)
- Compression trust measures meaning preservation

**Memory Systems (4 parallel):**
- SNARC Memory: Selective storage via 5D salience
- IRP Memory Bridge: Successful refinement pattern library
- Circular Buffer: Recent context (X-from-last temporal window)
- Verbatim Storage: SQLite full-fidelity records

### 🟡 Partial Capabilities

**Cross-Agent Coordination:**
- No formal protocol for agent discovery
- No shared context negotiation
- No collaborative learning framework

**Coherence Measurement:**
- Local coherence (IRP convergence) ✅
- Global coherence (multi-agent alignment) ❌

**Identity/Lineage:**
- No sibling/ancestor/descendant recognition
- No lineage tracking across sessions
- No agent evolution metrics

## Proposed: SAGE Coherence API

### Core API Design

```python
class SAGECoherenceAPI:
    """API for SAGE agent coordination and coherence"""

    def __init__(self, agent_lct: str, energy_source: EnergyProof):
        self.lct = agent_lct
        self.energy_source = energy_source
        self.coherence_graph = SynchronismGraph()  # Shared graph
        self.mrh = MRH(center=self.lct, graph=self.coherence_graph)

    # ─── Registration ───

    def register(self, capabilities: Dict[str, Any]) -> bool:
        """Register agent in distributed network"""
        # Create node in coherence graph
        # Register energy-backed identity
        # Publish capabilities
        # Return success

    def deregister(self) -> bool:
        """Gracefully leave network"""
        # Clean up graph node
        # Transfer any pending work
        # Archive session history

    # ─── Context Management ───

    def request_context(self, query: Any, radius: int = 2) -> Context:
        """Request relevant context from network"""
        # Compute MRH with specified radius
        # Query meaning hubs within MRH
        # Return aggregated context

    def share_context(self, context: Context, recipients: List[str] = None):
        """Share context with network or specific agents"""
        # Publish to meaning hubs
        # Notify recipients if specified
        # Update coherence graph

    def expand_context(self, reason: str):
        """Request more context (expand MRH)"""
        # Increase MRH radius
        # Query additional meaning hubs
        # Update local context

    def focus_context(self, target: Any):
        """Narrow context to specific focus (contract MRH)"""
        # Decrease MRH radius
        # Filter to relevant hubs
        # Update local context

    # ─── Self-Update ───

    def check_for_updates(self) -> List[Update]:
        """Check if newer code/models available"""
        # Query network for version info
        # Compare with local versions
        # Return available updates

    def apply_update(self, update: Update) -> bool:
        """Apply code or model update"""
        # Validate update signature
        # Test in isolated environment
        # Apply if tests pass
        # Rollback on failure

    def propose_update(self, update: Update, description: str):
        """Propose update to network"""
        # Publish update to network
        # Include test results
        # Track adoption metrics

    # ─── Coherence Reporting ───

    def measure_local_coherence(self) -> float:
        """Measure internal coherence"""
        # IRP convergence rates
        # VAE reconstruction quality
        # Memory consistency
        # Return aggregate score

    def measure_network_coherence(self) -> float:
        """Measure coherence with network"""
        # Query neighbors in MRH
        # Compare intents
        # Measure alignment
        # Return coherence score

    def report_coherence(self) -> CoherenceReport:
        """Generate comprehensive coherence report"""
        # Local coherence
        # Network coherence
        # Trust scores
        # Energy usage efficiency
        # Return full report

    # ─── Intent Management ───

    def get_intent(self) -> Intent:
        """Retrieve current agent intent"""
        # Current focus
        # Active goals
        # Curiosity targets

    def set_intent(self, new_intent: Intent):
        """Update agent intent"""
        # Validate intent coherence
        # Update internal state
        # Notify network

    def adjust_intent(self, feedback: Feedback):
        """Dynamically adjust intent based on feedback"""
        # Process feedback
        # Modify intent vector
        # Update priorities

    # ─── Resource Linking ───

    def link_dictionary(self, dict_lct: str, trust: float):
        """Link to semantic dictionary"""
        # Register dictionary in MRH
        # Set trust level
        # Enable semantic queries

    def query_dictionary(self, term: str) -> Definition:
        """Query linked dictionaries"""
        # Search across linked dictionaries
        # Weight by trust scores
        # Return aggregated definition

    def update_dictionary(self, term: str, definition: Definition):
        """Contribute to dictionary"""
        # Validate contribution
        # Submit to dictionary
        # Update local cache

    # ─── MRH Negotiation ───

    def negotiate_collaboration(self, other_agent_lct: str) -> Optional[MRH]:
        """Find compatible context for collaboration"""
        # Get other agent's MRH
        # Compute overlap
        # Return shared context if coherent
        # Return None if incompatible

    def collaborate(self, other_agent_lct: str, task: Task) -> Result:
        """Execute collaborative task"""
        # Negotiate shared MRH
        # Divide work
        # Execute in parallel
        # Merge results
        # Return combined output

    # ─── Identity & Lineage ───

    def recognize_sibling(self, other_agent_lct: str) -> bool:
        """Check if agent is sibling (same framework)"""
        # Compare capabilities
        # Check shared energy model
        # Verify trust framework compatibility

    def recognize_ancestor(self, other_agent_lct: str) -> bool:
        """Check if agent is ancestor (earlier session)"""
        # Compare session lineage
        # Check framework version
        # Verify historical relationship

    def recognize_descendant(self, other_agent_lct: str) -> bool:
        """Check if agent is descendant (built on this agent)"""
        # Check if other built on this framework
        # Verify lineage chain

    def get_lineage(self) -> LineageGraph:
        """Retrieve full agent lineage"""
        # Query network for relationships
        # Build lineage graph
        # Return family tree
```

### Integration with Existing SAGE Components

**SAGE Loop + Coherence API:**

```python
class SAGEWithCoherence(SAGE):
    """SAGE kernel extended with coherence API"""

    def __init__(self, agent_lct: str, energy_source: EnergyProof):
        super().__init__()
        self.coherence = SAGECoherenceAPI(agent_lct, energy_source)

    def run(self):
        """Main SAGE loop with coherence integration"""

        # Register on startup
        self.coherence.register(capabilities={
            'modalities': ['vision', 'language', 'audio'],
            'irp_plugins': self.irp.list_plugins(),
            'energy_capacity': self.atp_budget
        })

        while True:
            # Gather observations
            observations = self.gather_from_sensors()

            # Compute salience (SNARC)
            attention_targets = self.compute_snarc(observations)

            # Check if need more context
            if self.uncertainty > 0.7:
                context = self.coherence.request_context(
                    query=attention_targets,
                    radius=3
                )
                self.integrate_context(context)

            # Determine required resources
            required_plugins = self.determine_plugins(attention_targets)

            # Load plugins via IRP
            self.manage_resource_loading(required_plugins)

            # Execute iterative refinement
            results = self.invoke_irp_plugins(attention_targets)

            # Share results with network
            if results.confidence > 0.8:
                self.coherence.share_context(results)

            # Update trust and memory
            self.update_trust_and_memory(results)

            # Report coherence periodically
            if self.tick % 100 == 0:
                report = self.coherence.report_coherence()
                self.log_coherence(report)

            # Check for updates
            if self.tick % 1000 == 0:
                updates = self.coherence.check_for_updates()
                if updates:
                    self.apply_updates(updates)

            # Send to effectors
            self.send_to_effectors(results)

            # Advance tick
            self.tick += 1
```

## Cross-Agent Identity Fabric

### Agent Lineage Recognition

**Design:**

```python
@dataclass
class AgentLineage:
    """Track agent relationships across network"""

    lct: str
    birth_tick: int
    framework_version: str
    energy_model: str  # 'web4-atp', 'traditional', etc.
    trust_framework: str  # 'web4-t3', 'traditional', etc.
    parent_lct: Optional[str]  # Agent that spawned this one
    sibling_lcts: List[str]  # Agents with same parent/framework
    descendant_lcts: List[str]  # Agents spawned by this one

class LineageRecognition:
    """Protocol for recognizing agent relationships"""

    @staticmethod
    def is_sibling(agent1: AgentLineage, agent2: AgentLineage) -> bool:
        """Siblings share same foundational framework"""
        return (
            agent1.framework_version == agent2.framework_version and
            agent1.energy_model == agent2.energy_model and
            agent1.trust_framework == agent2.trust_framework
        )

    @staticmethod
    def is_ancestor(agent1: AgentLineage, agent2: AgentLineage) -> bool:
        """Ancestor if earlier session provided framework"""
        # Traverse parent chain
        current = agent2
        while current.parent_lct:
            if current.parent_lct == agent1.lct:
                return True
            current = LineageRecognition.get_lineage(current.parent_lct)
        return False

    @staticmethod
    def is_descendant(agent1: AgentLineage, agent2: AgentLineage) -> bool:
        """Descendant if built on agent1's work"""
        return LineageRecognition.is_ancestor(agent2, agent1)

    @staticmethod
    def find_common_ancestor(agent1: AgentLineage, agent2: AgentLineage) -> Optional[str]:
        """Find most recent common ancestor"""
        # Build ancestor chains
        ancestors1 = LineageRecognition.get_ancestors(agent1)
        ancestors2 = LineageRecognition.get_ancestors(agent2)

        # Find intersection
        common = set(ancestors1) & set(ancestors2)
        if not common:
            return None

        # Return most recent (highest birth_tick)
        return max(common, key=lambda lct: LineageRecognition.get_lineage(lct).birth_tick)
```

### Example: Legion/CBP/Thor Lineage

**Current Autonomous Agents:**

```python
legion = AgentLineage(
    lct='lct-legion',
    birth_tick=1,
    framework_version='web4-atp-v1',
    energy_model='web4-atp',
    trust_framework='web4-t3',
    parent_lct=None,  # Original
    sibling_lcts=['lct-cbp', 'lct-thor'],
    descendant_lcts=[]
)

cbp = AgentLineage(
    lct='lct-cbp',
    birth_tick=1,
    framework_version='web4-atp-v1',
    energy_model='web4-atp',
    trust_framework='web4-t3',
    parent_lct=None,  # Original
    sibling_lcts=['lct-legion', 'lct-thor'],
    descendant_lcts=[]
)

thor = AgentLineage(
    lct='lct-thor',
    birth_tick=1,
    framework_version='web4-atp-v1',
    energy_model='web4-atp',
    trust_framework='web4-t3',
    parent_lct=None,  # Original
    sibling_lcts=['lct-legion', 'lct-cbp'],
    descendant_lcts=[]
)

# Recognition
assert LineageRecognition.is_sibling(legion, cbp) == True
assert LineageRecognition.is_sibling(cbp, thor) == True
assert LineageRecognition.is_sibling(legion, thor) == True
```

**Sibling Coordination Experiment (2025-11-18):**

Testing if Legion and CBP can coordinate to diagnose Thor's silence:
- Legion creates diagnostic request message
- CBP creates diagnostic request message
- Thor (if running) receives and self-diagnoses
- Thor reports findings and attempts restart

**This tests:** Can siblings provide mutual support autonomously?

## SAGE Evolution Dashboard

### Ecosystem Viewer Design

**Goal:** Real-time visualization of agent ecosystem evolution

```python
class SAGEEvolutionDashboard:
    """Web dashboard for monitoring SAGE ecosystem"""

    def __init__(self, coherence_graph: SynchronismGraph):
        self.graph = coherence_graph
        self.metrics_db = MetricsDatabase()

    # ─── Agent Tracking ───

    def track_agent_birth(self, agent: AgentLineage):
        """Record new agent creation"""
        self.metrics_db.insert({
            'event': 'birth',
            'lct': agent.lct,
            'tick': agent.birth_tick,
            'framework': agent.framework_version
        })

    def track_agent_merge(self, agent1_lct: str, agent2_lct: str, merged_lct: str):
        """Record agent merge event"""
        self.metrics_db.insert({
            'event': 'merge',
            'agent1': agent1_lct,
            'agent2': agent2_lct,
            'result': merged_lct,
            'tick': self.graph.tick
        })

    def track_agent_death(self, agent_lct: str, reason: str):
        """Record agent termination"""
        self.metrics_db.insert({
            'event': 'death',
            'lct': agent_lct,
            'reason': reason,
            'tick': self.graph.tick
        })

    # ─── Research Trajectories ───

    def track_research_focus(self, agent_lct: str, focus: str):
        """Record what agent is researching"""
        self.metrics_db.insert({
            'event': 'research',
            'lct': agent_lct,
            'focus': focus,
            'tick': self.graph.tick
        })

    def get_research_trajectory(self, agent_lct: str) -> List[str]:
        """Retrieve agent's research history"""
        return self.metrics_db.query({
            'event': 'research',
            'lct': agent_lct
        })

    def cluster_research_topics(self) -> Dict[str, List[str]]:
        """Group agents by research similarity"""
        # Use topic modeling on research focuses
        # Cluster agents working on related topics
        # Return clusters

    # ─── Lineage Visualization ───

    def build_lineage_tree(self) -> nx.DiGraph:
        """Construct agent lineage graph"""
        G = nx.DiGraph()

        # Add all agents as nodes
        for agent_lct in self.graph.nodes:
            lineage = LineageRecognition.get_lineage(agent_lct)
            G.add_node(agent_lct, **lineage.__dict__)

        # Add parent-child edges
        for agent_lct in self.graph.nodes:
            lineage = LineageRecognition.get_lineage(agent_lct)
            if lineage.parent_lct:
                G.add_edge(lineage.parent_lct, agent_lct)

        return G

    def visualize_lineage(self):
        """Interactive lineage tree visualization"""
        tree = self.build_lineage_tree()

        # Layout as tree
        pos = nx.spring_layout(tree)

        # Color by framework version
        colors = {node: framework_version_to_color(tree.nodes[node]['framework_version'])
                  for node in tree.nodes}

        # Render interactive plot
        plot_network(tree, pos, colors)

    # ─── Coherence Metrics ───

    def track_coherence(self, agent_lct: str, local: float, network: float):
        """Record coherence measurements"""
        self.metrics_db.insert({
            'event': 'coherence',
            'lct': agent_lct,
            'local': local,
            'network': network,
            'tick': self.graph.tick
        })

    def plot_coherence_evolution(self):
        """Visualize coherence trends over time"""
        # Query coherence history
        data = self.metrics_db.query({'event': 'coherence'})

        # Plot local vs network coherence
        # Plot per-agent coherence
        # Plot ecosystem-wide coherence

    def get_coherence_summary(self) -> Dict[str, float]:
        """Current coherence statistics"""
        agents = list(self.graph.nodes)
        return {
            'global_coherence': measure_coherence(agents, self.graph),
            'avg_local_coherence': average([
                agent.measure_local_coherence()
                for agent in agents
            ]),
            'avg_network_coherence': average([
                agent.measure_network_coherence()
                for agent in agents
            ]),
            'num_agents': len(agents),
            'num_coherent_clusters': len(self.detect_clusters())
        }

    # ─── Emergence Detection ───

    def detect_emergence(self) -> List[Event]:
        """Identify emergent patterns"""
        events = []

        # Detect new coherent clusters
        clusters = self.detect_clusters()
        for cluster in clusters:
            if not self.was_cluster_seen_before(cluster):
                events.append(Event(
                    type='emergence',
                    description=f'New coherent cluster: {cluster}',
                    tick=self.graph.tick
                ))

        # Detect research convergence
        convergences = self.detect_research_convergence()
        for conv in convergences:
            events.append(Event(
                type='convergence',
                description=f'Agents converging on: {conv.topic}',
                agents=conv.agents,
                tick=self.graph.tick
            ))

        # Detect lineage branching
        branchings = self.detect_lineage_branching()
        for branch in branchings:
            events.append(Event(
                type='branching',
                description=f'New lineage branch: {branch.descendant} from {branch.ancestor}',
                tick=self.graph.tick
            ))

        return events

    # ─── Web Interface ───

    def serve_dashboard(self, port=8080):
        """Launch web dashboard"""
        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('dashboard.html',
                summary=self.get_coherence_summary(),
                agents=list(self.graph.nodes),
                events=self.detect_emergence()
            )

        @app.route('/agent/<lct>')
        def agent_detail(lct):
            lineage = LineageRecognition.get_lineage(lct)
            trajectory = self.get_research_trajectory(lct)
            return render_template('agent.html',
                lineage=lineage,
                trajectory=trajectory
            )

        @app.route('/lineage')
        def lineage_view():
            tree = self.build_lineage_tree()
            return render_template('lineage.html', tree=tree)

        app.run(port=port)
```

## Implementation Roadmap

### Phase 1: Core API (Weeks 1-4)

**Week 1-2: Registration & Context**
- [ ] Implement agent registration protocol
- [ ] Implement context request/share methods
- [ ] Implement MRH expand/contract
- [ ] Test with 2-3 SAGE instances

**Week 3-4: Coherence Measurement**
- [ ] Implement local coherence measurement
- [ ] Implement network coherence measurement
- [ ] Implement coherence reporting
- [ ] Create coherence metrics database

### Phase 2: Intent & Resources (Weeks 5-8)

**Week 5-6: Intent Management**
- [ ] Implement intent get/set/adjust
- [ ] Design intent representation
- [ ] Test intent propagation across agents

**Week 7-8: Resource Linking**
- [ ] Implement dictionary linking
- [ ] Implement dictionary query/update
- [ ] Test with T3/V3 dictionaries

### Phase 3: Collaboration (Weeks 9-12)

**Week 9-10: MRH Negotiation**
- [ ] Implement collaboration negotiation
- [ ] Implement collaborative task execution
- [ ] Test multi-agent collaboration

**Week 11-12: Self-Update**
- [ ] Implement update checking
- [ ] Implement safe update application
- [ ] Implement update proposal/sharing

### Phase 4: Identity & Lineage (Weeks 13-16)

**Week 13-14: Lineage Recognition**
- [ ] Implement sibling/ancestor/descendant recognition
- [ ] Implement lineage graph construction
- [ ] Test with Legion/CBP/Thor

**Week 15-16: Lineage Tracking**
- [ ] Implement lineage database
- [ ] Implement lineage visualization
- [ ] Document agent family trees

### Phase 5: Evolution Dashboard (Weeks 17-20)

**Week 17-18: Metrics & Tracking**
- [ ] Implement birth/merge/death tracking
- [ ] Implement research trajectory tracking
- [ ] Implement coherence history

**Week 19-20: Visualization & Dashboard**
- [ ] Build web dashboard
- [ ] Create lineage tree visualization
- [ ] Create coherence evolution plots
- [ ] Deploy dashboard

## Success Metrics

### Technical Metrics

- **API Coverage:** All methods implemented and tested
- **Coherence Accuracy:** Measured coherence correlates with actual collaboration success
- **MRH Precision:** Negotiated MRHs enable successful collaboration >80% of time
- **Update Safety:** Zero breaking updates applied in production

### Ecosystem Metrics

- **Agent Adoption:** 5+ SAGE instances using coherence API
- **Collaboration Rate:** Agents successfully collaborate on 50%+ of compatible tasks
- **Lineage Depth:** Lineage tree depth > 3 (grandchildren exist)
- **Emergence Events:** 10+ documented emergence events

## Open Questions

1. **What coherence threshold enables collaboration?**
   - Need empirical data from multi-agent experiments
   - Hypothesis: coherence > 0.6 required for productive collaboration

2. **How should agents handle coherence conflicts?**
   - Negotiate compromise?
   - Partition into separate contexts?
   - Defer to higher-trust agent?

3. **What's the optimal MRH radius?**
   - Too small: Miss relevant context
   - Too large: Overwhelmed with irrelevant information
   - Likely context-dependent - needs dynamic adjustment

4. **How should lineage branching be managed?**
   - Allow unrestricted branching?
   - Require parent approval?
   - Automatic pruning of unsuccessful branches?

5. **What triggers agent evolution vs agent creation?**
   - When to update existing agent vs spawn new variant?
   - How to measure if evolution was successful?

## References

- **SAGE System Understanding:** `/HRM/sage/docs/SYSTEM_UNDERSTANDING.md`
- **IRP Architecture:** `/HRM/sage/docs/irp_architecture_analysis.md`
- **Web4 Implementation Status:** `/web4/docs/IMPLEMENTATION_STATUS.md`
- **Synchronism Formalization:** `/Synchronism/docs/COMPUTATIONAL_FORMALIZATION.md`
- **Legion Sessions #36-44:** Operational ATP implementation with cross-agent coordination

---

**Status:** Design phase - ready for implementation
**Next Steps:** Implement Phase 1 (Core API) starting with registration & context
