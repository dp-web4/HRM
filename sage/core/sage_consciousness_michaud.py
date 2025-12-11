"""
SAGE Real Consciousness with Michaud Enhancements

Integrates André Michaud's neurolinguistic insights:
- AttentionManager for metabolic state-dependent resource allocation
- Enhanced memory consolidation based on satisfaction (energy minimization)
- Introspection-optimized for analytical conversations

This version addresses the quality degradation issues observed in basic loop by:
1. Dynamic attention allocation (WAKE/FOCUS/REST states)
2. Better ATP budget management
3. State-aware processing

Based on Michaud (2019): "The Mechanics of Conceptual Thinking"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import time

from sage.core.sage_consciousness import SAGEConsciousness
from sage.core.attention_manager import MetabolicState
from sage.core.mrh_aware_attention import MRHAwareAttentionManager
from sage.core.mrh_profile import MRHProfile, infer_mrh_profile_from_task
from sage.core.multimodal_atp_pricing import MultiModalATPPricer
from sage.core.quality_metrics import score_response_quality_normalized

# Session 31: Epistemic awareness integration
try:
    from sage.core.epistemic_states import estimate_epistemic_metrics
    EPISTEMIC_AWARENESS_AVAILABLE = True
except ImportError:
    EPISTEMIC_AWARENESS_AVAILABLE = False
from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory

# Phase 2.5: Federation integration (optional)
try:
    from sage.federation import (
        FederationRouter,
        FederationIdentity,
        FederationTask,
        ExecutionProof,
        SignedFederationTask,
        SignedExecutionProof,
        FederationKeyPair,
        FederationCrypto,
        SignatureRegistry,
        create_thor_identity,
        create_sprout_identity,
    )
    FEDERATION_AVAILABLE = True
except ImportError:
    FEDERATION_AVAILABLE = False


class MichaudSAGE(SAGEConsciousness):
    """
    SAGE Consciousness with Michaud's neurolinguistic enhancements.

    Key improvements over basic loop:
    - AttentionManager for metabolic state transitions
    - Dynamic ATP allocation based on salience and state
    - Enhanced memory consolidation (satisfaction-based)
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-Instruct",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        initial_atp: float = 100.0,
        irp_iterations: int = 3,
        salience_threshold: float = 0.15,
        attention_config: Optional[Dict] = None,
        # Session 26: Temporal adaptation parameters (optional)
        enable_temporal_adaptation: bool = False,
        temporal_adaptation_mode: str = "multi_objective",  # "production", "multi_objective", "conservative"
        # Phase 2.5: Federation parameters (optional)
        federation_enabled: bool = False,
        federation_identity: Optional['FederationIdentity'] = None,
        federation_platforms: Optional[list] = None,
        federation_key_path: Optional[str] = None
    ):
        """
        Initialize Michaud-enhanced SAGE consciousness.

        Args:
            model_path: Path to LLM (default: Introspective-Qwen for analytical tasks)
            base_model: Base model for LoRA adapters
            initial_atp: Initial ATP budget
            irp_iterations: IRP refinement iterations
            salience_threshold: SNARC salience threshold
            attention_config: Configuration for AttentionManager
            enable_temporal_adaptation: Enable automatic ATP parameter tuning (Session 26)
            temporal_adaptation_mode: Adaptation mode - "multi_objective" (recommended), "production", "conservative"
            federation_enabled: Enable federation routing (Phase 2.5+)
            federation_identity: Platform identity for federation
            federation_platforms: List of known platforms to register
            federation_key_path: Path to Ed25519 key file (auto-generated if None)
        """
        super().__init__(
            initial_atp=initial_atp,
            enable_circadian=False,  # Using AttentionManager instead
            simulation_mode=False
        )

        # Text input queue
        self.input_queue = deque(maxlen=100)

        # Michaud Enhancement #1: MRH-aware attention manager for metabolic states
        self.attention_manager = MRHAwareAttentionManager(
            total_atp=initial_atp,
            config=attention_config or {}
        )

        # Multi-modal ATP pricing for task cost calculation
        self.atp_pricer = MultiModalATPPricer()

        # Real LLM with optimized settings
        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model=base_model,
            max_history=2,  # Short history for stability
            irp_iterations=irp_iterations
        )

        # Lower temperature for analytical stability
        self.llm.llm.initial_temperature = 0.5
        self.llm.llm.min_temperature = 0.3

        # Real SNARC memory
        self.conversational_memory = ConversationalMemory(
            salience_threshold=salience_threshold
        )

        # Enhanced memory tracking
        self.satisfaction_history = []  # Track energy minimization success

        # Phase 2.5: Federation setup (optional)
        self.federation_enabled = federation_enabled
        self.federation_router = None
        self.federation_keypair = None
        self.signature_registry = None
        self.federation_identity = None

        if federation_enabled:
            if not FEDERATION_AVAILABLE:
                raise ImportError(
                    "Federation module not available. "
                    "Ensure sage.federation is installed."
                )

            # Auto-detect identity if not provided
            if federation_identity is None:
                federation_identity = self._detect_platform_identity()

            self.federation_identity = federation_identity

            # Initialize federation router
            self.federation_router = FederationRouter(federation_identity)

            # Generate or load Ed25519 key pair
            if federation_key_path is None:
                federation_key_path = f"sage/data/keys/{federation_identity.platform_name}_ed25519.key"

            self.federation_keypair = self._load_or_generate_keypair(
                federation_identity, federation_key_path
            )

            # Create signature registry and register self
            self.signature_registry = SignatureRegistry()
            self.signature_registry.register_platform(
                federation_identity.platform_name,
                self.federation_keypair.public_key_bytes()
            )

            # Register known platforms
            if federation_platforms:
                for platform in federation_platforms:
                    self.federation_router.register_platform(platform)
                    # In production, load public keys from LCT chain
                    # For Phase 2.5, we'll simulate this

            print(f"[Federation] Enabled:")
            print(f"  Platform: {federation_identity.platform_name}")
            print(f"  LCT ID: {federation_identity.lct_id}")
            print(f"  Known platforms: {len(federation_platforms) if federation_platforms else 0}")

        # Session 26: Temporal adaptation setup (optional)
        self.temporal_adaptation_enabled = enable_temporal_adaptation
        self.temporal_adapter = None
        self.high_salience_count = 0
        self.attended_high_salience = 0

        if enable_temporal_adaptation:
            from sage.core.temporal_adaptation import (
                create_production_adapter,
                create_multi_objective_adapter,
                create_conservative_adapter
            )

            # Create adapter based on mode
            if temporal_adaptation_mode == "multi_objective":
                # Session 25 validated: 3x energy efficiency, 12.2% fitness improvement
                self.temporal_adapter = create_multi_objective_adapter()
                print(f"[Temporal Adaptation] Using multi-objective (Session 25 recommended)")
            elif temporal_adaptation_mode == "production":
                self.temporal_adapter = create_production_adapter()
                print(f"[Temporal Adaptation] Using production mode")
            elif temporal_adaptation_mode == "conservative":
                self.temporal_adapter = create_conservative_adapter()
                print(f"[Temporal Adaptation] Using conservative mode")
            else:
                raise ValueError(f"Unknown temporal_adaptation_mode: {temporal_adaptation_mode}")

            # Get initial ATP parameters from adapter
            cost, recovery = self.temporal_adapter.get_current_params()
            print(f"[Temporal Adaptation] Initial parameters:")
            print(f"  Attention cost: {cost:.4f}")
            print(f"  Recovery rate: {recovery:.4f}")
            if hasattr(self.temporal_adapter, 'enable_multi_objective'):
                print(f"  Multi-objective: {self.temporal_adapter.enable_multi_objective}")
                if self.temporal_adapter.enable_multi_objective:
                    print(f"  Objective weights: coverage={self.temporal_adapter.coverage_weight:.0%}, "
                          f"quality={self.temporal_adapter.quality_weight:.0%}, "
                          f"energy={self.temporal_adapter.energy_weight:.0%}")

        print(f"[Michaud SAGE] Initialized with:")
        print(f"  Model: {model_path}")
        print(f"  Attention: {self.attention_manager}")
        print(f"  SNARC threshold: {salience_threshold}")
        print(f"  Temporal Adaptation: {'Enabled (' + temporal_adaptation_mode + ')' if enable_temporal_adaptation else 'Disabled'}")
        print(f"  Federation: {'Enabled' if federation_enabled else 'Disabled'}")

    def add_observation(self, text: str, context: Optional[Dict] = None):
        """Add text observation to input queue."""
        self.input_queue.append({
            'text': text,
            'context': context or {},
            'timestamp': time.time()
        })

    async def step(self):
        """
        Single consciousness cycle with Complete ATP Framework.

        Enhanced loop with MRH-aware horizon + multi-modal pricing:
        1. Gather observations
        2. Infer task properties (type, horizon, complexity)
        3. Calculate ATP cost (multi-modal pricing)
        4. Get ATP budget (MRH-aware, state-dependent)
        5. Resource decision: execute locally or route to federation
        6. MICHAUD: Update metabolic state based on salience
        7. MICHAUD: Allocate ATP via MRHAwareAttentionManager
        8. Execute IRP plugins with allocated resources
        9. MICHAUD: Update memory based on satisfaction (energy)
        10. Update trust weights
        """
        self.cycle_count += 1

        # 1. Gather observations
        observations = self._gather_observations()
        if not observations:
            return

        # 2. Infer task properties from first observation
        if observations:
            first_obs = observations[0]
            task_context = {
                'task_type': 'llm_inference',  # Language modality
                'operation': 'reason' if len(first_obs['data']) > 50 else 'recall',
                'session_length': self.attention_manager.get_time_in_state(),
                'num_agents': 1,
                'complexity': 'high' if len(first_obs['data']) > 100 else 'medium'
            }

            # Infer MRH horizon profile
            task_horizon = infer_mrh_profile_from_task(task_context)
            self.attention_manager.set_task_horizon(task_horizon)

            # 3. Calculate ATP cost (multi-modal pricing)
            complexity = task_context['complexity']
            estimated_latency = 30.0 if complexity == 'high' else 15.0  # seconds
            estimated_quality = 0.85 if complexity == 'high' else 0.90

            task_cost = self.atp_pricer.calculate_cost(
                task_type='llm_inference',
                complexity=complexity,
                latency=estimated_latency,
                quality=estimated_quality
            )

            # 4. Get ATP budget (MRH-aware)
            available_budget = self.attention_manager.get_total_allocated_atp(
                horizon=task_horizon
            )

            print(f"\n[ATP Framework] Cycle {self.cycle_count}")
            print(f"  Task: {task_context['operation']} ({complexity})")
            print(f"  Horizon: {task_horizon}")
            print(f"  Cost: {task_cost:.1f} ATP")
            print(f"  Budget: {available_budget:.1f} ATP")

            # 5. Resource decision with federation support (Phase 2.5)
            federation_delegated = False
            if task_cost > available_budget:
                # Insufficient ATP - check if state transition helps
                current_state = self.attention_manager.get_state()
                if current_state == MetabolicState.WAKE:
                    # Transition to FOCUS for more ATP
                    print(f"  Decision: Insufficient ATP in WAKE, transitioning to FOCUS")
                    self.attention_manager.current_state = MetabolicState.FOCUS
                    available_budget = self.attention_manager.get_total_allocated_atp(task_horizon)
                    print(f"  New budget (FOCUS): {available_budget:.1f} ATP")

                # Recheck after state transition
                if task_cost > available_budget:
                    # Still insufficient - try federation if enabled
                    if self.federation_enabled:
                        print(f"  Decision: Cost still exceeds budget, attempting federation")
                        federation_decision = self._handle_federation_routing(
                            task_context, task_cost, available_budget, task_horizon
                        )

                        if federation_decision['delegated']:
                            # Task successfully delegated
                            print(f"  ✓ Delegated to {federation_decision['platform']}")
                            federation_delegated = True
                            # Store federation results for use in result integration
                            # For now, we'll note it was delegated and continue
                        else:
                            print(f"  ✗ Federation failed: {federation_decision['reason']}")
                            print(f"  Decision: Executing with degradation")
                    else:
                        print(f"  Decision: Cost exceeds budget, executing with degradation")
                        print(f"  (Federation not enabled)")

            if not federation_delegated:
                print(f"  Decision: Execute locally ✓\n")

        # 6. Compute preliminary salience (before full execution)
        salience_map = {}
        for obs in observations:
            modality = obs['modality']
            # Simple heuristic: language is high priority
            if modality == 'language':
                salience_map[obs['id']] = 0.8
            else:
                salience_map[obs['id']] = 0.5

        # 7. MICHAUD: Update metabolic state and allocate ATP
        # MRHAwareAttentionManager automatically uses horizon scaling
        atp_allocation = self.attention_manager.allocate_attention_with_horizon(
            salience_map,
            horizon=task_horizon
        )

        current_state = self.attention_manager.get_state()
        print(f"[Michaud SAGE] State: {current_state.value}, "
              f"{self.attention_manager.get_time_in_state():.1f}s in state")

        # 8. Select and execute plugins
        plugins_to_execute = self._select_plugins(observations)

        # 9. Execute plugins with Michaud-enhanced processing
        results = {}
        for plugin_name in plugins_to_execute:
            obs_for_plugin = [o for o in observations
                            if self._plugin_handles_modality(plugin_name, o['modality'])]

            if obs_for_plugin:
                # Execute with ATP allocation
                allocated_atp = atp_allocation.get(obs_for_plugin[0]['id'], 0)
                result = await self._execute_plugin_michaud(
                    plugin_name,
                    obs_for_plugin[0],
                    allocated_atp,
                    task_horizon
                )
                results[plugin_name] = result

        # 10. MICHAUD: Update memory based on satisfaction
        self._update_memories_michaud(observations, results)

        # 11. Update trust weights
        self._update_trust_weights(results)

        # 12. Session 26: Update temporal adaptation (if enabled)
        if self.temporal_adaptation_enabled and self.temporal_adapter:
            self._update_temporal_adaptation(observations, results, salience_map, task_cost)

    async def _execute_plugin_michaud(
        self,
        plugin_name: str,
        observation: Dict,
        allocated_atp: float,
        task_horizon: MRHProfile
    ) -> Dict[str, Any]:
        """
        Execute plugin with Michaud-enhanced processing.

        Tracks satisfaction (energy minimization) for memory consolidation.
        Now includes horizon information for context.
        """
        if plugin_name == 'llm_reasoning':
            return await self._execute_llm_michaud(observation, allocated_atp, task_horizon)
        else:
            # Other plugins use standard execution
            return {'response': None, 'convergence_quality': 0.5}

    async def _execute_llm_michaud(
        self,
        observation: Dict,
        allocated_atp: float,
        task_horizon: MRHProfile
    ) -> Dict[str, Any]:
        """
        LLM execution with satisfaction tracking.

        Michaud: Satisfaction (low energy) drives memory consolidation.
        Now includes horizon-aware execution context.
        """
        question = observation['data']

        print(f"[LLM] Processing: \"{question[:50]}...\"")
        print(f"[LLM] ATP allocated: {allocated_atp:.2f}")
        print(f"[LLM] Task horizon: {task_horizon}")
        print(f"[LLM] Metabolic state: {self.attention_manager.get_state().value}")

        start_time = time.time()

        # Generate response with IRP refinement
        response, irp_info = self.llm.respond(question, use_irp=True)

        inference_time = time.time() - start_time

        # Michaud: Track satisfaction (energy minimization)
        initial_energy = irp_info.get('all_energies', [1.0])[0]
        final_energy = irp_info['final_energy']
        satisfaction = initial_energy - final_energy  # Energy reduction = satisfaction

        print(f"[LLM] Response generated ({inference_time:.2f}s)")
        print(f"[LLM] IRP: {irp_info['iterations']} iterations, "
              f"energy={final_energy:.3f}, satisfaction={satisfaction:.3f}")
        print(f"[LLM] Response: {response[:100]}...")

        return {
            'response': response,
            'irp_info': irp_info,
            'convergence_quality': 1.0 - final_energy,  # Lower energy = better
            'satisfaction': satisfaction,  # For Michaud-style consolidation
            'inference_time': inference_time
        }

    def _update_memories_michaud(
        self,
        observations: list,
        results: Dict[str, Any]
    ):
        """
        Michaud-enhanced memory consolidation.

        Key insight: Hippocampus strengthens patterns that provide satisfaction.
        SAGE equivalent: Consolidate experiences with high satisfaction (energy reduction).
        """
        for obs in observations:
            if obs['modality'] == 'language':
                question = obs['data']

                # Get LLM result if available
                llm_result = results.get('llm_reasoning', {})
                response = llm_result.get('response', '')
                irp_info = llm_result.get('irp_info', {})
                satisfaction = llm_result.get('satisfaction', 0.0)

                if response:
                    # Compute SNARC salience
                    is_salient, snarc_scores = self.conversational_memory.record_exchange(
                        question,
                        response,
                        irp_info
                    )

                    # Michaud: Store with satisfaction weight
                    if is_salient:
                        print(f"[SNARC] Salience: {snarc_scores['total_salience']:.3f} ✓ SALIENT")
                        print(f"[SNARC] Dimensions: "
                              f"S={snarc_scores['surprise']:.2f} "
                              f"N={snarc_scores['novelty']:.2f} "
                              f"A={snarc_scores['arousal']:.2f} "
                              f"R={snarc_scores['reward']:.2f} "
                              f"C={snarc_scores['conflict']:.2f}")

                        # Enhanced: Track satisfaction for pattern learning
                        self.satisfaction_history.append({
                            'cycle': self.cycle_count,
                            'salience': snarc_scores['total_salience'],
                            'satisfaction': satisfaction,
                            'final_energy': irp_info.get('final_energy', 1.0),
                            'metabolic_state': self.attention_manager.get_state().value
                        })

                    # Store to all memory systems
                    memory_entry = {
                        'cycle': self.cycle_count,
                        'question': question,
                        'response': response,
                        'salience': snarc_scores['total_salience'],
                        'satisfaction': satisfaction,  # Michaud enhancement
                        'irp_iterations': irp_info.get('iterations', 0),
                        'final_energy': irp_info.get('final_energy', 1.0),
                        'metabolic_state': self.attention_manager.get_state().value
                    }

                    # SNARC memory (selective)
                    if is_salient:
                        self.snarc_memory.append(memory_entry)

                    # Circular buffer (recent context)
                    self.circular_buffer.append(memory_entry)

    def _gather_observations(self):
        """Gather observations from input queue."""
        observations = []

        while self.input_queue:
            obs = self.input_queue.popleft()
            observations.append({
                'id': f"obs_{len(observations)}",
                'modality': 'language',
                'data': obs['text'],
                'context': obs.get('context', {}),
                'timestamp': obs['timestamp']
            })

        return observations

    def _select_plugins(self, observations):
        """Select plugins based on observation modalities."""
        plugins = set()

        for obs in observations:
            if obs['modality'] == 'language':
                plugins.add('llm_reasoning')

        return list(plugins)

    def _plugin_handles_modality(self, plugin_name: str, modality: str) -> bool:
        """Check if plugin handles modality."""
        return plugin_name == 'llm_reasoning' and modality == 'language'

    def _update_trust_weights(self, results: Dict[str, Any]):
        """Update plugin trust weights based on convergence quality."""
        for plugin_name, result in results.items():
            if isinstance(result, dict) and 'convergence_quality' in result:
                convergence_quality = result['convergence_quality']
                current_trust = self.plugin_trust_weights.get(plugin_name, 1.0)
                new_trust = 0.9 * current_trust + 0.1 * convergence_quality
                self.plugin_trust_weights[plugin_name] = new_trust

    def get_conversation_history(self):
        """Get conversation history for output."""
        return self.llm.get_history()

    def get_snarc_statistics(self):
        """Get SNARC memory statistics."""
        return self.conversational_memory.get_statistics()

    def get_attention_stats(self):
        """Get AttentionManager statistics."""
        return self.attention_manager.get_stats()

    def get_satisfaction_history(self):
        """Get satisfaction (energy minimization) history."""
        return self.satisfaction_history

    # Phase 2.5: Federation helper methods

    def _detect_platform_identity(self) -> 'FederationIdentity':
        """
        Auto-detect platform identity from hardware.

        Returns:
            FederationIdentity for this platform
        """
        # Try to detect from /proc/device-tree/model (Jetson)
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                if 'AGX Thor' in model:
                    return create_thor_identity()
                elif 'Orin Nano' in model:
                    return create_sprout_identity()
        except FileNotFoundError:
            pass

        # Default: create generic identity
        import socket
        from sage.core.mrh_profile import SpatialExtent, TemporalExtent, ComplexityExtent
        hostname = socket.gethostname()
        return FederationIdentity(
            platform_name=hostname,
            lct_id=f"{hostname}_sage_lct",
            hardware_spec=None,  # Unknown hardware
            supported_modalities=['llm'],
            max_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            stake=None
        )

    def _load_or_generate_keypair(
        self,
        identity: 'FederationIdentity',
        key_path: str
    ) -> 'FederationKeyPair':
        """
        Load existing key pair or generate new one.

        Args:
            identity: Platform identity
            key_path: Path to key file

        Returns:
            FederationKeyPair
        """
        from pathlib import Path

        key_file = Path(key_path)

        # Try to load existing key
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    private_key_bytes = f.read()
                keypair = FederationKeyPair.from_bytes(
                    identity.platform_name,
                    identity.lct_id,
                    private_key_bytes
                )
                print(f"[Federation] Loaded existing key from {key_path}")
                return keypair
            except Exception as e:
                print(f"[Federation] Warning: Could not load key ({e}), generating new one")

        # Generate new key pair
        keypair = FederationKeyPair.generate(
            identity.platform_name,
            identity.lct_id
        )

        # Save to disk
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, 'wb') as f:
            f.write(keypair.private_key_bytes())
        print(f"[Federation] Generated new key pair saved to {key_path}")

        return keypair

    def _create_federation_task(
        self,
        task_context: Dict[str, Any],
        task_cost: float,
        task_horizon: MRHProfile
    ) -> 'FederationTask':
        """
        Create FederationTask from consciousness loop context.

        Args:
            task_context: Task context dict
            task_cost: Estimated ATP cost
            task_horizon: MRH profile

        Returns:
            FederationTask ready for delegation
        """
        import uuid
        import time as time_module
        from sage.federation.federation_types import QualityRequirements

        # Get operation type and data
        task_type = task_context.get('operation', 'llm_inference')
        task_data = task_context.get('parameters', {})

        # Get quality requirements
        quality_reqs = task_context.get('quality', {})
        if isinstance(quality_reqs, dict):
            quality_requirements = QualityRequirements(
                min_quality=quality_reqs.get('min_quality', 0.7),
                min_convergence=quality_reqs.get('min_convergence', 0.6),
                max_energy=quality_reqs.get('max_energy', 0.7)
            )
        else:
            quality_requirements = quality_reqs

        return FederationTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            task_data=task_data,
            estimated_cost=task_cost,
            task_horizon=task_horizon,
            complexity=task_context.get('complexity', 'medium'),
            delegating_platform=self.federation_identity.platform_name,
            delegating_state=self.attention_manager.get_state(),
            quality_requirements=quality_requirements,
            max_latency=300.0,  # 5 minutes max latency
            deadline=time_module.time() + 3600.0,  # 1 hour deadline
            min_witnesses=3
        )

    def _handle_federation_routing(
        self,
        task_context: Dict[str, Any],
        task_cost: float,
        local_budget: float,
        task_horizon: MRHProfile
    ) -> Dict[str, Any]:
        """
        Handle federation routing decision and execution (Phase 2.5).

        Args:
            task_context: Task context from consciousness loop
            task_cost: Estimated ATP cost
            local_budget: Available local ATP budget
            task_horizon: MRH profile

        Returns:
            {
                'delegated': bool,
                'platform': str | None,
                'reason': str,
                'results': Dict | None
            }
        """
        # Create federation task
        task = self._create_federation_task(task_context, task_cost, task_horizon)

        # Check if should delegate
        should_delegate, reason = self.federation_router.should_delegate(
            task, local_budget
        )

        if not should_delegate:
            return {
                'delegated': False,
                'platform': None,
                'reason': reason,
                'results': None
            }

        # Find capable platforms
        candidates = self.federation_router.find_capable_platforms(task)
        if not candidates:
            return {
                'delegated': False,
                'platform': None,
                'reason': 'no_capable_platforms',
                'results': None
            }

        # Select best platform (highest reputation)
        target_platform = candidates[0]

        # Phase 2.5: Simulated delegation (no network)
        # Phase 3 will replace this with actual gRPC call
        try:
            execution_proof = self._simulate_federation_delegation(task, target_platform)

            # Validate proof
            if not self._validate_execution_proof(execution_proof, task):
                return {
                    'delegated': False,
                    'platform': target_platform.platform_name,
                    'reason': 'proof_validation_failed',
                    'results': None
                }

            # Update reputation
            self.federation_router.update_platform_reputation(
                target_platform.lct_id,
                execution_proof.quality_score,
                execution_proof.actual_cost
            )

            # Return successful delegation
            return {
                'delegated': True,
                'platform': target_platform.platform_name,
                'reason': 'federation_success',
                'results': execution_proof.result_data
            }

        except Exception as e:
            print(f"[Federation] Error during delegation: {e}")
            return {
                'delegated': False,
                'platform': target_platform.platform_name,
                'reason': f'delegation_error: {e}',
                'results': None
            }

    def _simulate_federation_delegation(
        self,
        task: 'FederationTask',
        target_platform: 'FederationIdentity'
    ) -> 'ExecutionProof':
        """
        Simulate federation delegation for Phase 2.5.

        In Phase 3, this will be replaced with actual gRPC network call.

        Args:
            task: Federation task to delegate
            target_platform: Platform to delegate to

        Returns:
            ExecutionProof from simulated execution
        """
        print(f"\n[FEDERATION SIMULATION]")
        print(f"  Delegating to: {target_platform.platform_name}")
        print(f"  Task: {task.task_type}")
        print(f"  Cost estimate: {task.estimated_cost:.1f} ATP")

        # Simulate: Remote platform executes task
        # For Phase 2.5, we just simulate success with quality ~0.75
        simulated_result_data = {
            'status': 'completed',
            'output': f'Simulated result from {target_platform.platform_name}',
            'simulated': True
        }

        # Simulate actual cost (slightly lower than estimate for good platform)
        actual_cost = task.estimated_cost * 0.9
        actual_latency = 15.0  # Simulated 15 second execution

        # Simulate quality metrics (good platform)
        quality_score = 0.75
        final_energy = 0.25  # Lower energy = better
        convergence_quality = 0.8
        irp_iterations = 3

        # Create execution proof
        execution_proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform=target_platform.platform_name,
            result_data=simulated_result_data,
            actual_latency=actual_latency,
            actual_cost=actual_cost,
            irp_iterations=irp_iterations,
            final_energy=final_energy,
            convergence_quality=convergence_quality,
            quality_score=quality_score
        )

        print(f"  Execution: Complete ✓")
        print(f"  Quality: {quality_score:.2f}")
        print(f"  Actual cost: {actual_cost:.1f} ATP\n")

        return execution_proof

    def _validate_execution_proof(
        self,
        proof: 'ExecutionProof',
        task: 'FederationTask'
    ) -> bool:
        """
        Validate execution proof from federation.

        Args:
            proof: Execution proof to validate
            task: Original task

        Returns:
            True if proof is valid
        """
        # Basic validation
        if proof.task_id != task.task_id:
            print(f"[Federation] Proof validation failed: task_id mismatch")
            return False

        if proof.quality_score < 0.0 or proof.quality_score > 1.0:
            print(f"[Federation] Proof validation failed: invalid quality score")
            return False

        if proof.actual_cost < 0:
            print(f"[Federation] Proof validation failed: negative cost")
            return False

        # In Phase 2.5, we trust simulated proofs
        # In Phase 3, this will verify Ed25519 signatures
        return True

    def _update_temporal_adaptation(
        self,
        observations: List[Dict],
        results: Dict,
        salience_map: Dict[str, float],
        task_cost: float
    ):
        """
        Update temporal adaptation based on cycle performance.

        Session 26: Integrate multi-objective temporal adaptation into production.
        Tracks coverage, quality, and energy efficiency to automatically tune ATP parameters.

        Args:
            observations: Observations processed this cycle
            results: Plugin execution results
            salience_map: Salience scores for observations
            task_cost: ATP cost of primary task
        """
        # Extract performance metrics from this cycle

        # Did we attend to observations? (allocated ATP > 0)
        attended = len(results) > 0

        # Get mean salience across observations
        if salience_map:
            mean_salience = sum(salience_map.values()) / len(salience_map)
        else:
            mean_salience = 0.5

        # Track high-salience observations (salience > 0.7)
        for sal in salience_map.values():
            if sal > 0.7:
                self.high_salience_count += 1
                if attended:
                    self.attended_high_salience += 1

        # Get current ATP level (normalized to 0-1)
        atp_level = self.atp / 100.0

        # Extract quality score from LLM results (if available)
        # Session 27: Use 4-metric quality scoring instead of convergence_quality proxy
        quality_score = None
        convergence_iterations = 3  # Default
        if 'llm_reasoning' in results:
            llm_result = results['llm_reasoning']
            # Get actual response text for quality scoring
            response_text = llm_result.get('response', None)
            convergence_iterations = llm_result.get('iterations', 3)
            if response_text:
                # Score using 4-metric system: unique, technical, numbers, no hedging
                quality_score = score_response_quality_normalized(response_text)

                # Session 31: Estimate epistemic metrics alongside quality
                if EPISTEMIC_AWARENESS_AVAILABLE:
                    epistemic_metrics = estimate_epistemic_metrics(
                        response_text=response_text,
                        quality_score=quality_score,
                        convergence_iterations=convergence_iterations,
                        salience=mean_salience
                    )
                    # Track epistemic state in temporal adapter
                    self.temporal_adapter.update_epistemic_state(epistemic_metrics)
            else:
                # Fallback to convergence_quality if response text unavailable
                quality_score = llm_result.get('convergence_quality', None)

        # Update temporal adapter
        adaptation_result = self.temporal_adapter.update(
            attended=attended,
            salience=mean_salience,
            atp_level=atp_level,
            high_salience_count=self.high_salience_count,
            attended_high_salience=self.attended_high_salience,
            quality_score=quality_score,
            attention_cost=task_cost
        )

        # If adaptation occurred, log it
        if adaptation_result is not None:
            new_cost, new_recovery = adaptation_result
            print(f"\n[Temporal Adaptation] ATP parameters updated:")
            print(f"  Cost: {new_cost:.4f}")
            print(f"  Recovery: {new_recovery:.4f}")

            # Get current metrics
            metrics = self.temporal_adapter.current_window.get_metrics()
            print(f"  Current metrics:")
            print(f"    Coverage: {metrics.get('coverage', 0):.1%}")
            if 'quality' in metrics:
                print(f"    Quality: {metrics.get('quality', 0):.1%}")
            if 'energy_efficiency' in metrics:
                print(f"    Energy: {metrics.get('energy_efficiency', 0):.1%}")
            if 'weighted_fitness' in metrics:
                print(f"    Weighted Fitness: {metrics.get('weighted_fitness', 0):.3f}")

            # Get adaptation statistics
            stats = self.temporal_adapter.get_statistics()
            print(f"  Total adaptations: {stats['total_adaptations']}")
            print(f"  Current damping: {stats['current_damping']:.2f}x")

    def get_temporal_adaptation_stats(self) -> Dict:
        """
        Get temporal adaptation statistics.

        Session 26: Production monitoring of temporal adaptation performance.

        Returns:
            Dictionary with adaptation metrics, or empty dict if disabled
        """
        if not self.temporal_adaptation_enabled or not self.temporal_adapter:
            return {}

        stats = self.temporal_adapter.get_statistics()

        # Add current metrics
        metrics = self.temporal_adapter.current_window.get_metrics()
        stats.update({
            'current_coverage': metrics.get('coverage', 0),
            'current_quality': metrics.get('quality', 0),
            'current_energy': metrics.get('energy_efficiency', 0),
            'current_fitness': metrics.get('weighted_fitness', 0),
            'high_salience_count': self.high_salience_count,
            'attended_high_salience': self.attended_high_salience
        })

        return stats

    def __repr__(self):
        stats = self.get_snarc_statistics()
        attention_stats = self.get_attention_stats()
        return (f"MichaudSAGE(cycle={self.cycle_count}, "
                f"state={attention_stats['current_state']}, "
                f"snarc_captured={stats.get('salient_exchanges', 0)}, "
                f"avg_salience={stats.get('avg_salience', 0):.3f})")
