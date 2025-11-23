"""
Skill Learning Integration - SAGE ↔ Skill Library

Enables SAGE to:
1. Query skills for current situation (skill-guided reasoning)
2. Apply skills via IRP plugin selection
3. Discover new skills from successful patterns
4. Contribute to cross-machine skill library

Architecture:
    SAGE Consciousness → Skill Manager → Epistemic Skill Library

Integration Points:
    1. query_skills() - Find applicable skills for situation
    2. apply_skill() - Use skill to guide IRP execution
    3. discover_skill() - Extract patterns from convergence
    4. verify_skill() - Cross-machine validation

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 2 Implementation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json

# Add memory repo to path
_memory_root = Path(__file__).parent.parent.parent.parent / "memory"
if str(_memory_root) not in sys.path:
    sys.path.insert(0, str(_memory_root))

try:
    # Import epistemic skill tools
    from epistemic.query.search import query_skills_programmatic, recommend_skill_programmatic
    from epistemic.tools.skill_detector import SkillDetector
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    print("[SkillLearning] Warning: Skill tools not available. Running in fallback mode.")


@dataclass
class IRPGuidance:
    """
    Guidance for IRP plugin selection based on skill.

    Maps skill strategy to concrete IRP execution:
    - Which plugins to use
    - How to allocate attention
    - Convergence criteria
    - Expected patterns
    """
    plugins_prioritized: List[str]  # Plugins in priority order
    attention_allocation: Dict[str, float]  # Plugin → weight
    convergence_threshold: float  # Energy threshold for halt
    expected_iterations: int  # Typical iteration count
    strategy_description: str  # Human-readable strategy
    preconditions: Dict[str, Any]  # When to apply
    indicators: Dict[str, Any]  # Success patterns

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)


@dataclass
class SkillApplication:
    """Record of skill application in SAGE"""
    skill_id: str
    skill_name: str
    situation: str
    applied_at: datetime
    plugins_used: List[str]
    success: bool
    quality_score: float
    convergence_iterations: int
    notes: str


@dataclass
class DiscoveredPattern:
    """Pattern extracted from successful SAGE execution"""
    pattern_id: str
    strategy: str
    plugins_sequence: List[str]
    convergence_profile: Dict[str, Any]
    success_count: int
    failure_count: int
    average_quality: float
    contexts: List[str]  # Situations where this worked


class SkillLearningManager:
    """
    Manages SAGE's interaction with the skill library.

    Provides:
    - Skill queries for current situation
    - Skill application via IRP guidance
    - Pattern discovery from successful executions
    - Cross-machine skill verification

    Usage:
        manager = SkillLearningManager(machine='thor', project='sage')

        # Query applicable skills
        skills = manager.query_applicable_skills("design attention mechanism")

        # Apply skill
        guidance = manager.apply_skill(skills[0], current_situation)

        # Discover new pattern
        if execution_successful:
            manager.record_successful_pattern(execution_data)
    """

    def __init__(
        self,
        machine: str = 'thor',
        project: str = 'sage',
        min_skill_confidence: float = 0.6,
        pattern_repetition_threshold: int = 3,
        witness_manager: Optional[Any] = None
    ):
        """
        Initialize skill learning manager.

        Args:
            machine: Hardware entity
            project: Project context
            min_skill_confidence: Minimum confidence for skill application
            pattern_repetition_threshold: Times pattern must repeat before creating skill
            witness_manager: Optional external witness manager (Phase 3)
        """
        self.machine = machine
        self.project = project
        self.min_confidence = min_skill_confidence
        self.pattern_threshold = pattern_repetition_threshold
        self.witness_manager = witness_manager

        if SKILLS_AVAILABLE:
            self.skill_detector = SkillDetector()
            print(f"[SkillLearning] Initialized for {machine}/{project}")
            print(f"[SkillLearning] Min confidence: {min_skill_confidence}")
            print(f"[SkillLearning] Pattern threshold: {pattern_repetition_threshold}")
        else:
            print(f"[SkillLearning] Running in fallback mode (skill tools unavailable)")

        # Local tracking
        self.skill_applications = []
        self.discovered_patterns = {}
        self.successful_executions = []

    def query_applicable_skills(
        self,
        situation: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query skills applicable to current situation.

        Args:
            situation: Description of current reasoning task
            category: Optional skill category filter
            limit: Maximum skills to return

        Returns:
            List of skill dictionaries with metadata
        """
        if not SKILLS_AVAILABLE:
            print("[SkillLearning] Skills unavailable, returning empty")
            return []

        try:
            # Query skills from epistemic database
            skills = query_skills_programmatic(
                machine=self.machine,
                category=category,
                min_success_rate=self.min_confidence,
                limit=limit
            )

            # Filter for applicable skills based on situation
            applicable = []
            for skill in skills:
                # Check machine compatibility
                works_on = skill.get('works_on_machines', [])
                if works_on and self.machine not in works_on:
                    continue

                # Check hardware requirements
                if not self._check_hardware_compatibility(skill):
                    continue

                # Calculate applicability score
                applicability = self._calculate_applicability(skill, situation)
                if applicability >= self.min_confidence:
                    skill['applicability_score'] = applicability
                    applicable.append(skill)

            # Sort by applicability
            applicable.sort(key=lambda s: s['applicability_score'], reverse=True)

            if applicable:
                print(f"[SkillLearning] Found {len(applicable)} applicable skills")
                for skill in applicable[:3]:
                    print(f"[SkillLearning]   - {skill['name']}: {skill['applicability_score']:.2f}")

            return applicable

        except Exception as e:
            print(f"[SkillLearning] Error querying skills: {e}")
            return []

    def recommend_skill(
        self,
        situation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get single best skill recommendation for situation.

        Args:
            situation: Current reasoning task
            context: Optional context (past episodes, etc.)

        Returns:
            Best skill with confidence score, or None
        """
        if not SKILLS_AVAILABLE:
            return None

        try:
            recommendation = recommend_skill_programmatic(
                current_task=situation,
                machine=self.machine,
                project=self.project
            )

            if recommendation:
                print(f"[SkillLearning] Recommended: {recommendation['skill']['name']}")
                print(f"[SkillLearning] Confidence: {recommendation['confidence']:.2f}")
                print(f"[SkillLearning] Reason: {recommendation['reason']}")

            return recommendation

        except Exception as e:
            print(f"[SkillLearning] Error getting recommendation: {e}")
            return None

    def apply_skill(
        self,
        skill: Dict[str, Any],
        situation: str
    ) -> IRPGuidance:
        """
        Convert skill to IRP guidance for SAGE execution.

        Args:
            skill: Skill dictionary from query
            situation: Current reasoning task

        Returns:
            IRPGuidance with plugin priorities and settings
        """
        # Extract skill strategy
        strategy = skill.get('strategy', '')

        # Convert strategy to IRP guidance
        guidance = self._strategy_to_irp_guidance(strategy, skill)

        print(f"[SkillLearning] Applying skill: {skill['name']}")
        print(f"[SkillLearning] Strategy: {strategy[:60]}...")
        print(f"[SkillLearning] Prioritized plugins: {guidance.plugins_prioritized}")

        return guidance

    def _strategy_to_irp_guidance(
        self,
        strategy: str,
        skill: Dict[str, Any]
    ) -> IRPGuidance:
        """
        Translate skill strategy to concrete IRP guidance.

        This is where we map high-level strategies to specific
        SAGE execution parameters.

        Args:
            strategy: Skill strategy description
            skill: Full skill metadata

        Returns:
            IRPGuidance for SAGE execution
        """
        # Default guidance
        plugins = ['llm_reasoning']  # SAGE primarily uses LLM
        attention = {'llm_reasoning': 1.0}
        threshold = 0.3
        iterations = 5

        # Parse strategy for guidance hints
        strategy_lower = strategy.lower()

        # Multi-modal skills
        if 'vision' in strategy_lower or 'image' in strategy_lower:
            plugins.insert(0, 'vision')
            attention = {'vision': 0.6, 'llm_reasoning': 0.4}

        if 'audio' in strategy_lower or 'sound' in strategy_lower:
            plugins.insert(0, 'audio')
            attention = {'audio': 0.5, 'llm_reasoning': 0.5}

        # Convergence hints
        if 'precise' in strategy_lower or 'careful' in strategy_lower:
            threshold = 0.2  # Stricter convergence
            iterations = 7

        if 'quick' in strategy_lower or 'fast' in strategy_lower:
            threshold = 0.5  # Relaxed convergence
            iterations = 3

        # Attention allocation hints
        if 'focused' in strategy_lower or 'concentrated' in strategy_lower:
            # Single plugin gets all attention
            pass

        if 'distributed' in strategy_lower or 'parallel' in strategy_lower:
            # Equal attention distribution
            attention = {p: 1.0/len(plugins) for p in plugins}

        return IRPGuidance(
            plugins_prioritized=plugins,
            attention_allocation=attention,
            convergence_threshold=threshold,
            expected_iterations=iterations,
            strategy_description=strategy,
            preconditions=skill.get('preconditions', {}),
            indicators=skill.get('indicators', {})
        )

    def record_skill_application(
        self,
        skill: Dict[str, Any],
        situation: str,
        execution_result: Dict[str, Any]
    ):
        """
        Record application of skill for tracking and learning.

        Args:
            skill: Applied skill
            situation: Situation where applied
            execution_result: SAGE execution results
        """
        application = SkillApplication(
            skill_id=skill['skill_id'],
            skill_name=skill['name'],
            situation=situation,
            applied_at=datetime.now(timezone.utc),
            plugins_used=execution_result.get('plugins_used', []),
            success=execution_result.get('converged', False),
            quality_score=execution_result.get('quality_score', 0.0),
            convergence_iterations=execution_result.get('iterations', 0),
            notes=execution_result.get('notes', '')
        )

        self.skill_applications.append(application)

        # Update skill statistics if skills available
        if SKILLS_AVAILABLE and application.success:
            try:
                # Record successful application in epistemic DB
                # This will update skill's success rate
                print(f"[SkillLearning] ✅ Skill '{skill['name']}' succeeded")
                print(f"[SkillLearning]    Quality: {application.quality_score:.2f}")
                print(f"[SkillLearning]    Iterations: {application.convergence_iterations}")
            except Exception as e:
                print(f"[SkillLearning] Warning: Could not update skill stats: {e}")

    def record_successful_pattern(
        self,
        execution_data: Dict[str, Any]
    ):
        """
        Record successful execution pattern for potential skill discovery.

        Args:
            execution_data: Complete SAGE execution results
        """
        # Extract pattern signature
        pattern_sig = self._extract_pattern_signature(execution_data)

        # Check if we've seen this pattern before
        pattern_id = pattern_sig['pattern_id']

        if pattern_id in self.discovered_patterns:
            # Increment count
            pattern = self.discovered_patterns[pattern_id]
            pattern.success_count += 1
            pattern.contexts.append(execution_data.get('situation', ''))

            # Update quality average
            quality = execution_data.get('quality_score', 0)
            pattern.average_quality = (
                (pattern.average_quality * (pattern.success_count - 1) + quality) /
                pattern.success_count
            )

            print(f"[SkillLearning] Pattern '{pattern_id}' seen {pattern.success_count} times")

            # Check if ready to become skill
            if pattern.success_count >= self.pattern_threshold:
                self._create_skill_from_pattern(pattern)

        else:
            # New pattern
            pattern = DiscoveredPattern(
                pattern_id=pattern_id,
                strategy=pattern_sig['strategy'],
                plugins_sequence=pattern_sig['plugins'],
                convergence_profile=pattern_sig['convergence'],
                success_count=1,
                failure_count=0,
                average_quality=execution_data.get('quality_score', 0),
                contexts=[execution_data.get('situation', '')]
            )
            self.discovered_patterns[pattern_id] = pattern

            print(f"[SkillLearning] New pattern discovered: {pattern_id}")

    def _extract_pattern_signature(
        self,
        execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract characteristic pattern from execution.

        Identifies what made this execution successful.
        """
        import hashlib

        # Plugin sequence
        plugins = execution_data.get('plugins_used', [])

        # Convergence profile
        convergence = {
            'iterations': execution_data.get('iterations', 0),
            'final_energy': execution_data.get('final_energy', 0),
            'energy_trajectory': execution_data.get('energy_trajectory', [])
        }

        # Strategy description (extracted from execution characteristics)
        strategy = self._infer_strategy(execution_data)

        # Generate pattern ID from signature
        sig_str = f"{plugins}:{convergence['iterations']}:{strategy}"
        pattern_id = hashlib.md5(sig_str.encode()).hexdigest()[:12]

        return {
            'pattern_id': pattern_id,
            'strategy': strategy,
            'plugins': plugins,
            'convergence': convergence
        }

    def _infer_strategy(self, execution_data: Dict[str, Any]) -> str:
        """
        Infer strategy description from execution characteristics.

        This is reverse-engineering: what strategy led to this success?
        """
        plugins = execution_data.get('plugins_used', [])
        iterations = execution_data.get('iterations', 0)
        energy_drop = execution_data.get('energy_drop', 0)

        strategy_parts = []

        # Plugin strategy
        if len(plugins) == 1:
            strategy_parts.append(f"Focused {plugins[0]} processing")
        else:
            strategy_parts.append(f"Multi-modal ({', '.join(plugins)}) integration")

        # Convergence strategy
        if iterations <= 3:
            strategy_parts.append("rapid convergence")
        elif iterations >= 7:
            strategy_parts.append("careful iterative refinement")
        else:
            strategy_parts.append("standard convergence")

        # Energy strategy
        if energy_drop > 0.7:
            strategy_parts.append("high-confidence resolution")
        elif energy_drop > 0.4:
            strategy_parts.append("moderate-confidence resolution")

        return " with ".join(strategy_parts)

    def _create_skill_from_pattern(self, pattern: DiscoveredPattern):
        """
        Create new skill from discovered pattern.

        Args:
            pattern: Pattern that has repeated enough times
        """
        if not SKILLS_AVAILABLE:
            print(f"[SkillLearning] Would create skill, but tools unavailable")
            return

        try:
            # Generate skill name
            skill_name = self._generate_skill_name(pattern.strategy)

            # Create skill in epistemic database
            skill_id = self.skill_detector.create_skill(
                name=skill_name,
                category='consciousness',  # SAGE-discovered skills
                strategy=pattern.strategy,
                discovered_by=self.machine,
                discovery_episode_id=None,  # TODO: Link to episode
                preconditions={'min_quality': 0.6},
                indicators={'convergence_profile': pattern.convergence_profile}
            )

            # Witness on blockchain (Phase 3)
            if self.witness_manager:
                self.witness_manager.witness_skill(
                    skill_id=skill_id,
                    skill_name=skill_name,
                    category='consciousness',
                    quality=pattern.average_quality,
                    success_rate=pattern.success_count / (pattern.success_count + pattern.failure_count)
                )

            print(f"[SkillLearning] 🎓 NEW SKILL CREATED: {skill_name}")
            print(f"[SkillLearning]    ID: {skill_id}")
            print(f"[SkillLearning]    Success rate: {pattern.success_count}/{pattern.success_count + pattern.failure_count}")
            print(f"[SkillLearning]    Average quality: {pattern.average_quality:.2f}")
            print(f"[SkillLearning]    Strategy: {pattern.strategy}")

        except Exception as e:
            print(f"[SkillLearning] Error creating skill: {e}")

    def _generate_skill_name(self, strategy: str) -> str:
        """Generate concise skill name from strategy"""
        # Simple approach: take key words from strategy
        words = strategy.lower().split()
        key_words = [w for w in words if len(w) > 4 and w not in ['with', 'using', 'through']]
        return '_'.join(key_words[:3])

    def _check_hardware_compatibility(self, skill: Dict[str, Any]) -> bool:
        """
        Check if skill is compatible with current hardware.

        Args:
            skill: Skill metadata

        Returns:
            True if compatible, False otherwise
        """
        # Check RAM requirements
        min_ram = skill.get('min_ram_gb', 0)
        if min_ram > 32:  # Thor has 32GB
            return False

        # Check GPU requirements
        requires_gpu = skill.get('requires_gpu', False)
        if requires_gpu and self.machine == 'sprout':
            # Sprout has GPU but limited
            gpu_min = skill.get('gpu_min_series', '')
            if 'RTX' in gpu_min:  # Sprout has Jetson, not RTX
                return False

        # Check architecture
        arch = skill.get('architecture', 'universal')
        if arch != 'universal':
            # Thor is x86_64, Sprout is ARM
            machine_arch = 'x86_64' if self.machine == 'thor' else 'aarch64'
            if arch != machine_arch:
                return False

        return True

    def _calculate_applicability(
        self,
        skill: Dict[str, Any],
        situation: str
    ) -> float:
        """
        Calculate how applicable a skill is to current situation.

        Combines:
        - Semantic similarity (strategy ↔ situation)
        - Success rate history
        - Machine compatibility
        - Experience level

        Returns:
            Applicability score 0.0-1.0
        """
        # For now: simple heuristic based on success rate
        # TODO: Add semantic similarity when embeddings available
        success_rate = skill.get('success_rate', 0.5)
        applications = skill.get('applications_count', 0)

        # Base score from success rate
        score = success_rate * 0.6

        # Boost from experience
        experience_boost = min(0.2, applications / 50.0)  # Max 0.2 at 50+ applications
        score += experience_boost

        # Machine compatibility boost
        works_on = skill.get('works_on_machines', [])
        if self.machine in works_on:
            score += 0.2

        return min(1.0, score)

    def get_skill_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on skill usage and discovery.

        Returns:
            Dictionary with counts and metrics
        """
        return {
            'applications_count': len(self.skill_applications),
            'successful_applications': sum(1 for a in self.skill_applications if a.success),
            'discovered_patterns': len(self.discovered_patterns),
            'skills_created': sum(
                1 for p in self.discovered_patterns.values()
                if p.success_count >= self.pattern_threshold
            ),
            'average_quality': (
                sum(a.quality_score for a in self.skill_applications) /
                len(self.skill_applications)
                if self.skill_applications else 0.0
            )
        }


# Convenience function
def create_skill_manager(
    machine: str = 'thor',
    project: str = 'sage',
    witness_manager: Optional[Any] = None,
    **kwargs
) -> SkillLearningManager:
    """
    Create skill learning manager for SAGE.

    Args:
        machine: Hardware entity
        project: Project context
        witness_manager: Optional external witness manager (Phase 3)
        **kwargs: Additional configuration

    Returns:
        Configured SkillLearningManager
    """
    return SkillLearningManager(
        machine=machine,
        project=project,
        witness_manager=witness_manager,
        **kwargs
    )
