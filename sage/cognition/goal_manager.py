#!/usr/bin/env python3
"""
Goal Manager for SAGE SNARC Cognition
======================================

Maintains hierarchical goals, tracks progress, manages goal activation/inhibition.

Track 3: SNARC Cognition - Component 4/4
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import uuid

# Import from Track 2 (Memory)
try:
    from sage.memory.retrieval import MemoryRetrieval
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.memory.retrieval import MemoryRetrieval

# Import from Track 3 (Cognition)
try:
    from sage.cognition.working_memory import WorkingMemory
except ModuleNotFoundError:
    from working_memory import WorkingMemory


@dataclass
class Goal:
    """
    Goal in the goal hierarchy

    Examples:
    - "Explore environment" (high-level)
    - "Navigate to landmark" (mid-level)
    - "Avoid obstacle" (low-level)
    """
    goal_id: str
    description: str
    goal_type: str  # "exploration", "navigation", "manipulation", etc.

    # Hierarchy
    parent_goal: Optional[str] = None  # Parent goal ID
    subgoals: List[str] = field(default_factory=list)  # Child goal IDs

    # Activation
    activation: float = 0.0  # 0.0-1.0 (how active is this goal?)
    priority: float = 0.5  # 0.0-1.0 (how important?)

    # Progress
    progress: float = 0.0  # 0.0-1.0 (how close to completion?)
    status: str = "pending"  # "pending", "active", "blocked", "completed", "failed"

    # Success criteria
    success_condition: Optional[Callable[[Dict[str, Any]], bool]] = None

    # Context
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate goal"""
        if not 0.0 <= self.activation <= 1.0:
            raise ValueError(f"activation must be 0-1, got {self.activation}")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be 0-1, got {self.priority}")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"progress must be 0-1, got {self.progress}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'goal_type': self.goal_type,
            'parent_goal': self.parent_goal,
            'subgoals': self.subgoals,
            'activation': self.activation,
            'priority': self.priority,
            'progress': self.progress,
            'status': self.status,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'metadata': self.metadata
        }


@dataclass
class GoalSwitchEvent:
    """Record of goal switching"""
    timestamp: float
    from_goal: Optional[str]
    to_goal: str
    reason: str  # "blocked", "completed", "higher_priority", "interrupt"
    switching_cost: float  # Time or resources consumed by switch


class GoalManager:
    """
    Hierarchical goal management system

    Design:
    - DAG structure (directed acyclic graph) for goal hierarchy
    - Spreading activation for goal activation/inhibition
    - Progress tracking (leaf goals manual, parent goals computed)
    - Goal switching with context preservation

    Integration:
    - Attention Manager: Active goals guide attention
    - Deliberation Engine: Active goals inform planning
    - Working Memory: Goal context stored in working memory
    - Track 2 (Memory): Learn successful goal sequences
    """

    def __init__(
        self,
        memory_retrieval: Optional[MemoryRetrieval] = None,
        working_memory: Optional[WorkingMemory] = None,
        max_active_goals: int = 3
    ):
        """
        Initialize Goal Manager

        Args:
            memory_retrieval: Memory system from Track 2
            working_memory: Working memory from Track 3
            max_active_goals: Maximum number of simultaneously active goals
        """
        self.memory = memory_retrieval
        self.working_memory = working_memory
        self.max_active_goals = max_active_goals

        # Goal registry
        self.goals: Dict[str, Goal] = {}

        # Goal hierarchy (DAG)
        self.hierarchy = nx.DiGraph()

        # Active goals (goal_ids)
        self.active_goals: List[str] = []

        # Goal history
        self.goal_history: List[GoalSwitchEvent] = []

        # Statistics
        self.total_goals_created: int = 0
        self.total_goals_completed: int = 0
        self.total_goals_failed: int = 0
        self.total_switches: int = 0

        # Activation spreading parameters
        self.parent_decay: float = 0.8  # Activation decay when spreading to parent
        self.child_decay: float = 0.9  # Activation decay when spreading to children
        self.conflict_inhibition: float = 0.5  # Inhibition for conflicting goals

    def add_goal(
        self,
        goal: Goal,
        parent_goal_id: Optional[str] = None
    ) -> str:
        """
        Add goal to hierarchy

        Args:
            goal: Goal to add
            parent_goal_id: Parent goal (None for root goal)

        Returns:
            goal_id of added goal
        """
        # Validate parent exists
        if parent_goal_id is not None and parent_goal_id not in self.goals:
            raise ValueError(f"Parent goal {parent_goal_id} not found")

        # Check for cycles (would create invalid DAG)
        if parent_goal_id is not None:
            # Would adding this edge create a cycle?
            self.hierarchy.add_edge(parent_goal_id, goal.goal_id)
            if not nx.is_directed_acyclic_graph(self.hierarchy):
                self.hierarchy.remove_edge(parent_goal_id, goal.goal_id)
                raise ValueError(f"Adding goal {goal.goal_id} would create cycle in hierarchy")
            self.hierarchy.remove_edge(parent_goal_id, goal.goal_id)  # Remove test edge

        # Add to registry
        goal.parent_goal = parent_goal_id
        self.goals[goal.goal_id] = goal
        self.total_goals_created += 1

        # Add to hierarchy
        self.hierarchy.add_node(goal.goal_id)
        if parent_goal_id is not None:
            self.hierarchy.add_edge(parent_goal_id, goal.goal_id)
            # Update parent's subgoal list
            self.goals[parent_goal_id].subgoals.append(goal.goal_id)

        # Add to working memory if high priority
        if goal.priority >= 0.7 and self.working_memory:
            self.working_memory.add_item(
                content_type='goal',
                content=goal,
                priority=goal.priority,
                goal_id=goal.goal_id
            )

        return goal.goal_id

    def activate_goal(self, goal_id: str):
        """
        Activate a goal (and spread activation to related goals)

        Spreading activation:
        - Parents receive activation * parent_decay
        - Children receive activation * child_decay
        - Conflicting goals receive inhibition

        Args:
            goal_id: Goal to activate
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")

        goal = self.goals[goal_id]

        # Set activation
        goal.activation = 1.0
        goal.status = "active"
        goal.last_updated = time.time()

        # Add to active goals if not present
        if goal_id not in self.active_goals:
            self.active_goals.append(goal_id)

        # Spread activation to parents (recursively)
        if goal.parent_goal:
            self._spread_activation_to_parent(goal.parent_goal, self.parent_decay)

        # Spread activation to children
        for child_id in goal.subgoals:
            self._spread_activation_to_child(child_id, self.child_decay)

        # Inhibit conflicting goals
        self._inhibit_conflicting_goals(goal_id)

        # Add to working memory
        if self.working_memory:
            self.working_memory.add_item(
                content_type='goal',
                content=goal,
                priority=goal.priority,
                goal_id=goal_id
            )

    def deactivate_goal(self, goal_id: str):
        """
        Deactivate a goal (and its descendants)

        Args:
            goal_id: Goal to deactivate
        """
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]

        # Set activation
        goal.activation = 0.0
        if goal.status == "active":
            goal.status = "pending"
        goal.last_updated = time.time()

        # Remove from active goals
        if goal_id in self.active_goals:
            self.active_goals.remove(goal_id)

        # Deactivate children recursively
        for child_id in goal.subgoals:
            self.deactivate_goal(child_id)

        # Clear from working memory
        if self.working_memory:
            self.working_memory.clear(goal_id=goal_id)

    def update_progress(
        self,
        goal_id: str,
        progress_delta: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Update goal progress

        For leaf goals: Manual progress updates
        For parent goals: Computed from subgoal progress

        Args:
            goal_id: Goal to update
            progress_delta: Change in progress (can be negative)
            context: Additional context
        """
        if goal_id not in self.goals:
            return

        goal = self.goals[goal_id]

        # Update leaf goal
        if not goal.subgoals:
            goal.progress = max(0.0, min(1.0, goal.progress + progress_delta))
        else:
            # Recompute parent progress from subgoals
            goal.progress = self._compute_parent_progress(goal_id)

        goal.last_updated = time.time()

        # Check completion
        if goal.progress >= 1.0:
            goal.status = "completed"
            self.total_goals_completed += 1
            self.deactivate_goal(goal_id)

        # Update parent progress recursively
        if goal.parent_goal:
            self.update_progress(goal.parent_goal, 0.0)  # Trigger recomputation

    def get_active_goals(
        self,
        filter_by: Optional[str] = None
    ) -> List[Goal]:
        """
        Get currently active goals

        Args:
            filter_by: Filter by goal_type (None = all active goals)

        Returns:
            List of active goals
        """
        active = [self.goals[gid] for gid in self.active_goals if gid in self.goals]

        if filter_by:
            active = [g for g in active if g.goal_type == filter_by]

        # Sort by activation * priority
        active.sort(key=lambda g: g.activation * g.priority, reverse=True)

        return active

    def check_goal_completion(
        self,
        goal_id: str,
        situation: Dict[str, Any]
    ) -> bool:
        """
        Check if goal is satisfied

        Args:
            goal_id: Goal to check
            situation: Current situation

        Returns:
            True if goal is satisfied
        """
        if goal_id not in self.goals:
            return False

        goal = self.goals[goal_id]

        # Use success condition if provided
        if goal.success_condition:
            return goal.success_condition(situation)

        # Fallback: Check progress
        return goal.progress >= 1.0

    def handle_goal_conflict(
        self,
        goal1_id: str,
        goal2_id: str
    ) -> str:
        """
        Resolve conflict between two goals

        Strategy:
        - Compare priorities
        - Consider progress (don't abandon nearly-complete goals)
        - Check resource constraints

        Args:
            goal1_id: First goal
            goal2_id: Second goal

        Returns:
            goal_id of winner
        """
        if goal1_id not in self.goals or goal2_id not in self.goals:
            return goal1_id if goal1_id in self.goals else goal2_id

        goal1 = self.goals[goal1_id]
        goal2 = self.goals[goal2_id]

        # Compute conflict scores
        score1 = self._compute_goal_score(goal1)
        score2 = self._compute_goal_score(goal2)

        # Winner
        winner_id = goal1_id if score1 >= score2 else goal2_id
        loser_id = goal2_id if winner_id == goal1_id else goal1_id

        # Inhibit loser
        self.goals[loser_id].activation *= self.conflict_inhibition

        return winner_id

    def suggest_next_goal(
        self,
        situation: Dict[str, Any]
    ) -> Optional[Goal]:
        """
        Suggest next goal to pursue

        Based on:
        - Current situation
        - Goal priorities
        - Memory of successful goal sequences
        - Progress on existing goals

        Args:
            situation: Current situation

        Returns:
            Suggested goal (or None)
        """
        # Get candidate goals (not active, not completed/failed)
        candidates = [
            g for g in self.goals.values()
            if g.goal_id not in self.active_goals
            and g.status not in ["completed", "failed"]
        ]

        if not candidates:
            return None

        # Score candidates
        scores = {}
        for goal in candidates:
            score = self._compute_goal_score(goal)

            # Boost score based on memory of successful sequences
            if self.memory:
                # TODO: Query memory for successful goal sequences
                # For now, just use intrinsic score
                pass

            scores[goal.goal_id] = score

        # Select highest-scoring goal
        best_goal_id = max(scores, key=scores.get)
        return self.goals[best_goal_id]

    def switch_goal(
        self,
        from_goal_id: Optional[str],
        to_goal_id: str,
        reason: str
    ) -> float:
        """
        Switch from one goal to another

        Switching cost:
        - Save context to working memory
        - Deactivate old goal
        - Activate new goal

        Args:
            from_goal_id: Current goal (None if no current goal)
            to_goal_id: New goal
            reason: Reason for switch

        Returns:
            Switching cost (seconds)
        """
        start_time = time.time()

        # Save old goal context to working memory
        if from_goal_id and from_goal_id in self.goals:
            old_goal = self.goals[from_goal_id]
            if self.working_memory:
                self.working_memory.add_item(
                    content_type='goal',
                    content=old_goal,
                    priority=old_goal.priority * 0.7,  # Reduced priority for suspended goal
                    goal_id=from_goal_id
                )
            self.deactivate_goal(from_goal_id)

        # Activate new goal
        self.activate_goal(to_goal_id)

        # Record switch
        switching_cost = time.time() - start_time
        switch_event = GoalSwitchEvent(
            timestamp=time.time(),
            from_goal=from_goal_id,
            to_goal=to_goal_id,
            reason=reason,
            switching_cost=switching_cost
        )
        self.goal_history.append(switch_event)
        self.total_switches += 1

        return switching_cost

    def _spread_activation_to_parent(self, parent_id: str, activation: float):
        """Spread activation to parent goal (recursive)"""
        if parent_id not in self.goals:
            return

        parent = self.goals[parent_id]
        parent.activation = max(parent.activation, activation)
        parent.last_updated = time.time()

        # Continue spreading to grandparent
        if parent.parent_goal:
            self._spread_activation_to_parent(
                parent.parent_goal,
                activation * self.parent_decay
            )

    def _spread_activation_to_child(self, child_id: str, activation: float):
        """Spread activation to child goal"""
        if child_id not in self.goals:
            return

        child = self.goals[child_id]
        child.activation = max(child.activation, activation)
        child.last_updated = time.time()

    def _inhibit_conflicting_goals(self, goal_id: str):
        """
        Inhibit goals that conflict with the given goal

        For now, simple heuristic: Inhibit all other active goals
        (can be refined with explicit conflict declarations)
        """
        for other_id in self.active_goals:
            if other_id != goal_id:
                self.goals[other_id].activation *= self.conflict_inhibition

    def _compute_parent_progress(self, parent_id: str) -> float:
        """Compute progress of parent goal from subgoals"""
        parent = self.goals[parent_id]

        if not parent.subgoals:
            return parent.progress

        # Mean progress of subgoals
        subgoal_progress = [
            self.goals[sid].progress
            for sid in parent.subgoals
            if sid in self.goals
        ]

        if not subgoal_progress:
            return 0.0

        return np.mean(subgoal_progress)

    def _compute_goal_score(self, goal: Goal) -> float:
        """
        Compute goal score for selection

        Score = priority * activation * (1 + progress_bonus)

        Where progress_bonus boosts nearly-complete goals
        """
        progress_bonus = goal.progress * 0.5  # Up to 50% boost
        score = goal.priority * goal.activation * (1.0 + progress_bonus)
        return score

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID"""
        return self.goals.get(goal_id)

    def get_goal_hierarchy(self, root_goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get goal hierarchy as nested dictionary

        Args:
            root_goal_id: Start from this goal (None = all roots)

        Returns:
            Nested dictionary representation
        """
        if root_goal_id:
            return self._build_hierarchy_tree(root_goal_id)

        # Get all root goals (no parents)
        roots = [g for g in self.goals.values() if g.parent_goal is None]

        return {
            'roots': [self._build_hierarchy_tree(r.goal_id) for r in roots]
        }

    def _build_hierarchy_tree(self, goal_id: str) -> Dict[str, Any]:
        """Build hierarchy tree from goal"""
        if goal_id not in self.goals:
            return {}

        goal = self.goals[goal_id]

        return {
            'goal': goal.to_dict(),
            'children': [self._build_hierarchy_tree(c) for c in goal.subgoals]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get goal manager statistics"""
        return {
            'total_goals': len(self.goals),
            'active_goals': len(self.active_goals),
            'total_created': self.total_goals_created,
            'total_completed': self.total_goals_completed,
            'total_failed': self.total_goals_failed,
            'total_switches': self.total_switches,
            'hierarchy_depth': nx.dag_longest_path_length(self.hierarchy) if self.hierarchy.nodes else 0,
            'avg_activation': np.mean([g.activation for g in self.goals.values()]) if self.goals else 0.0,
            'avg_progress': np.mean([g.progress for g in self.goals.values()]) if self.goals else 0.0
        }


def test_goal_manager():
    """Test Goal Manager implementation"""
    print("\n" + "="*60)
    print("TESTING GOAL MANAGER")
    print("="*60)

    # Create goal manager
    gm = GoalManager(max_active_goals=3)

    # Test 1: Add goals
    print("\n1. Adding goals to hierarchy...")

    # Root goal
    explore_goal = Goal(
        goal_id="explore",
        description="Explore the environment",
        goal_type="exploration",
        priority=0.8
    )
    gm.add_goal(explore_goal)

    # Subgoals
    nav_goal = Goal(
        goal_id="navigate",
        description="Navigate to landmark",
        goal_type="navigation",
        priority=0.9
    )
    gm.add_goal(nav_goal, parent_goal_id="explore")

    avoid_goal = Goal(
        goal_id="avoid",
        description="Avoid obstacles",
        goal_type="navigation",
        priority=0.7
    )
    gm.add_goal(avoid_goal, parent_goal_id="navigate")

    stats = gm.get_stats()
    print(f"   Added {stats['total_goals']} goals")
    print(f"   Hierarchy depth: {stats['hierarchy_depth']}")

    # Test 2: Activate goal
    print("\n2. Activating goal...")
    gm.activate_goal("navigate")
    active = gm.get_active_goals()
    print(f"   Active goals: {len(active)}")
    print(f"   Top active: {active[0].goal_id} (activation={active[0].activation:.2f})")

    # Test 3: Update progress
    print("\n3. Updating progress...")
    gm.update_progress("avoid", 0.5)
    avoid = gm.get_goal("avoid")
    print(f"   avoid progress: {avoid.progress:.1%}")

    gm.update_progress("avoid", 0.5)
    print(f"   avoid progress after update: {avoid.progress:.1%}")
    print(f"   avoid status: {avoid.status}")

    # Test 4: Goal switching
    print("\n4. Testing goal switching...")
    search_goal = Goal(
        goal_id="search",
        description="Search for objects",
        goal_type="exploration",
        priority=0.95
    )
    gm.add_goal(search_goal, parent_goal_id="explore")

    cost = gm.switch_goal("navigate", "search", reason="higher_priority")
    print(f"   Switched from navigate → search")
    print(f"   Switching cost: {cost*1000:.2f}ms")

    active = gm.get_active_goals()
    print(f"   New active goal: {active[0].goal_id}")

    # Test 5: Goal conflict
    print("\n5. Testing goal conflict resolution...")
    winner = gm.handle_goal_conflict("navigate", "search")
    print(f"   Conflict winner: {winner}")
    winner_goal = gm.get_goal(winner)
    print(f"   Winner activation: {winner_goal.activation:.2f}")

    # Test 6: Suggest next goal
    print("\n6. Suggesting next goal...")
    gm.deactivate_goal("search")
    situation = {'location': [0, 0], 'obstacles': []}
    suggestion = gm.suggest_next_goal(situation)
    if suggestion:
        print(f"   Suggested: {suggestion.goal_id} - {suggestion.description}")
        print(f"   Priority: {suggestion.priority:.2f}")

    # Test 7: Goal hierarchy
    print("\n7. Getting goal hierarchy...")
    hierarchy = gm.get_goal_hierarchy()
    print(f"   Root goals: {len(hierarchy['roots'])}")
    if hierarchy['roots']:
        root = hierarchy['roots'][0]
        print(f"   Root: {root['goal']['description']}")
        print(f"   Children: {len(root['children'])}")

    # Test 8: Statistics
    print("\n8. Goal manager statistics...")
    final_stats = gm.get_stats()
    print(f"   Total goals: {final_stats['total_goals']}")
    print(f"   Active goals: {final_stats['active_goals']}")
    print(f"   Completed: {final_stats['total_completed']}")
    print(f"   Switches: {final_stats['total_switches']}")
    print(f"   Avg progress: {final_stats['avg_progress']:.1%}")

    print("\n" + "="*60)
    print("✅ GOAL MANAGER TESTS PASSED")
    print("="*60)

    return gm


if __name__ == "__main__":
    test_goal_manager()
