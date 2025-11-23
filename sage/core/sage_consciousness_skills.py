"""
SAGE Consciousness with Skill Integration

Extends EpistemicSAGEConsciousness to integrate with skill library:
- Query applicable skills before reasoning
- Apply skill-guided IRP execution
- Discover new skills from patterns
- Contribute to cross-machine skill library

This is SAGE with skill-accelerated learning.

Author: Thor (SAGE Development Platform)
Date: 2025-11-22
Status: Phase 2 Implementation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
from datetime import datetime, timezone

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness_epistemic import EpistemicSAGEConsciousness
from integration.skill_learning import (
    SkillLearningManager,
    IRPGuidance,
    create_skill_manager
)


class SkillSAGEConsciousness(EpistemicSAGEConsciousness):
    """
    SAGE Consciousness with skill library integration.

    Extensions over Epistemic SAGE:
    - Queries skills before reasoning
    - Applies skill-guided plugin selection
    - Discovers new skills from successful patterns
    - Tracks skill applications

    The consciousness loop gains skill-accelerated learning.
    """

    def __init__(
        self,
        machine: str = 'thor',
        project: str = 'sage',
        enable_skill_discovery: bool = True,
        enable_skill_application: bool = True,
        **kwargs
    ):
        """
        Initialize SAGE with skill integration.

        Args:
            machine: Hardware entity
            project: Project context
            enable_skill_discovery: Discover new skills from patterns
            enable_skill_application: Apply skills from library
            **kwargs: Additional args for EpistemicSAGEConsciousness
        """
        # Create witness manager (Phase 3)
        from integration.witness_manager import create_witness_manager
        witness_manager = create_witness_manager(
            machine=machine,
            project=project,
            enable_witnessing=kwargs.get('enable_witnessing', True)
        )

        # Initialize epistemic consciousness with witness manager
        super().__init__(
            machine=machine,
            project=project,
            witness_manager=witness_manager,
            **kwargs
        )

        # Initialize skill manager with witness manager
        print(f"[Skills] Initializing skill learning...")
        self.skill_manager = create_skill_manager(
            machine=machine,
            project=project,
            witness_manager=witness_manager
        )

        self.enable_skill_discovery = enable_skill_discovery
        self.enable_skill_application = enable_skill_application

        # Skill tracking for current session
        self.session_skills_applied = []
        self.session_skills_discovered = []
        self.session_patterns_recorded = []

        print(f"[Skills] ✅ Skill integration complete")
        print(f"[Skills] Discovery: {'enabled' if enable_skill_discovery else 'disabled'}")
        print(f"[Skills] Application: {'enabled' if enable_skill_application else 'disabled'}")
        print()

    def start_session(self, session_id: Optional[str] = None):
        """Override to reset skill tracking"""
        super().start_session(session_id)

        self.session_skills_applied = []
        self.session_skills_discovered = []
        self.session_patterns_recorded = []

    async def _execute_plugins(
        self,
        attention_targets: List,
        budget_allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Execute plugins with skill integration.

        Extensions:
        1. Query applicable skills for situation
        2. Apply skill guidance to plugin selection
        3. Record successful patterns
        4. Discover new skills
        """
        # Extract situation from language observations
        situation = None
        for target in attention_targets:
            if target.observation.modality == 'language':
                situation = target.observation.data['text']
                break

        # Phase 1: Skill-guided execution (if enabled and situation exists)
        skill_guidance = None
        applied_skill = None

        if self.enable_skill_application and situation:
            # Query applicable skills
            print(f"\n[Skills] Querying skills for situation...")
            applicable_skills = self.skill_manager.query_applicable_skills(
                situation,
                limit=3
            )

            if applicable_skills:
                # Use highest-rated skill
                best_skill = applicable_skills[0]
                print(f"[Skills] Applying skill: {best_skill['name']}")
                print(f"[Skills] Applicability: {best_skill.get('applicability_score', 0):.2f}")
                print(f"[Skills] Success rate: {best_skill.get('success_rate', 0):.2f}")

                # Get IRP guidance from skill
                skill_guidance = self.skill_manager.apply_skill(best_skill, situation)
                applied_skill = best_skill

                # Modify budget allocation based on skill
                if skill_guidance:
                    self._apply_skill_guidance(budget_allocation, skill_guidance)

        # Phase 2: Execute plugins (potentially skill-guided)
        results = await super()._execute_plugins(attention_targets, budget_allocation)

        # Phase 3: Post-execution skill learning (if enabled)
        if self.enable_skill_discovery and results:
            for plugin_name, result in results.items():
                if result.get('convergence_quality', 0) > 0.7:  # High quality result
                    # Record successful pattern
                    execution_data = {
                        'situation': situation or '',
                        'plugins_used': [plugin_name],
                        'iterations': result.get('irp_info', {}).get('iterations', 0),
                        'final_energy': result.get('irp_info', {}).get('final_energy', 1.0),
                        'quality_score': result.get('convergence_quality', 0),
                        'energy_drop': 1.0 - result.get('irp_info', {}).get('final_energy', 1.0),
                        'energy_trajectory': result.get('irp_info', {}).get('energy_history', [])
                    }

                    self.skill_manager.record_successful_pattern(execution_data)
                    self.session_patterns_recorded.append(execution_data)

        # Phase 4: Track skill application (if skill was used)
        if applied_skill and results:
            # Extract convergence info from first result
            first_result = next(iter(results.values()))

            execution_result = {
                'plugins_used': list(results.keys()),
                'converged': first_result.get('convergence_quality', 0) > 0.5,
                'quality_score': first_result.get('convergence_quality', 0),
                'iterations': first_result.get('irp_info', {}).get('iterations', 0),
                'notes': f"Skill-guided execution"
            }

            self.skill_manager.record_skill_application(
                applied_skill,
                situation,
                execution_result
            )

            self.session_skills_applied.append(applied_skill['name'])

        return results

    def _apply_skill_guidance(
        self,
        budget_allocation: Dict[str, float],
        guidance: IRPGuidance
    ):
        """
        Modify ATP budget allocation based on skill guidance.

        Args:
            budget_allocation: Current budget (modified in-place)
            guidance: Skill-provided guidance
        """
        # Adjust budget based on skill's attention allocation
        total_budget = sum(budget_allocation.values())

        for plugin, weight in guidance.attention_allocation.items():
            if plugin in budget_allocation:
                # Adjust existing allocation
                budget_allocation[plugin] = total_budget * weight
            else:
                # Add new plugin allocation
                budget_allocation[plugin] = total_budget * weight * 0.5  # Partial

    def end_session(self) -> Optional[str]:
        """
        Override to include skill statistics in session summary.
        """
        episode_id = super().end_session()

        # Display skill statistics
        if self.session_skills_applied or self.session_patterns_recorded:
            print()
            print(f"[Skills] Session skill activity:")
            print(f"[Skills]   Skills applied: {len(self.session_skills_applied)}")
            if self.session_skills_applied:
                for skill_name in set(self.session_skills_applied):
                    count = self.session_skills_applied.count(skill_name)
                    print(f"[Skills]     - {skill_name} ({count}x)")

            print(f"[Skills]   Patterns recorded: {len(self.session_patterns_recorded)}")

            # Get skill manager statistics
            stats = self.skill_manager.get_skill_statistics()
            if stats['skills_created'] > 0:
                print(f"[Skills]   🎓 New skills created: {stats['skills_created']}")

        return episode_id

    async def run_with_skill_learning(
        self,
        observations: List[str],
        session_id: Optional[str] = None
    ):
        """
        Run consciousness loop with skill learning enabled.

        This is the main entry point for skill-integrated SAGE.

        Args:
            observations: List of text observations
            session_id: Optional session ID

        Returns:
            Episode ID and skill statistics
        """
        # Start session
        self.start_session(session_id)

        try:
            # Add observations
            for obs in observations:
                self.add_observation(obs)

            # Run consciousness loop
            print(f"[Consciousness] Starting skill-integrated loop...")
            print(f"[Consciousness] {len(observations)} observations to process")
            print()

            # Process each observation
            for i in range(len(observations)):
                if self.input_queue:
                    await self.cycle()
                    print()

            # End session
            episode_id = self.end_session()

            # Get final skill statistics
            skill_stats = self.skill_manager.get_skill_statistics()

            return {
                'episode_id': episode_id,
                'skills_applied': len(self.session_skills_applied),
                'patterns_recorded': len(self.session_patterns_recorded),
                'total_skill_applications': skill_stats['applications_count'],
                'total_skills_created': skill_stats['skills_created'],
                'average_quality': skill_stats['average_quality']
            }

        except Exception as e:
            print(f"[Session] Error during run: {e}")
            import traceback
            traceback.print_exc()

            self.end_session()
            raise


# Convenience function
def create_skill_sage(
    machine: str = 'thor',
    project: str = 'sage',
    **kwargs
) -> SkillSAGEConsciousness:
    """
    Create SAGE consciousness with full skill integration.

    Args:
        machine: Hardware entity
        project: Project context
        **kwargs: Additional configuration

    Returns:
        Configured SkillSAGEConsciousness instance
    """
    return SkillSAGEConsciousness(
        machine=machine,
        project=project,
        **kwargs
    )
