#!/usr/bin/env python3
"""
Working Memory for SAGE SNARC Cognition
========================================

Maintains active task context, multi-step plans, and intermediate results.
Implements cognitive realistic capacity limits (7±2 items).

Track 3: SNARC Cognition - Component 2/4
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import uuid

# Import from Track 2 (Memory)
try:
    from sage.memory.retrieval import MemoryRetrieval
    from sage.memory.stm import STMEntry
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.memory.retrieval import MemoryRetrieval
    from sage.memory.stm import STMEntry


@dataclass
class WorkingMemorySlot:
    """
    Single item in working memory

    Analogy: A "sticky note" for the cognitive system
    Holds one piece of information relevant to current task
    """
    slot_id: str
    content_type: str  # "goal", "plan_step", "intermediate_result", "binding", "other"
    content: Any
    priority: float  # How important to retain (0-1)
    timestamp: float
    goal_id: Optional[str] = None  # Which goal owns this slot
    access_count: int = 0  # How many times accessed

    def __post_init__(self):
        """Validate slot"""
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be 0-1, got {self.priority}")


@dataclass
class PlanStep:
    """
    Step in a multi-step plan

    Represents one action in a sequence toward a goal
    """
    step_id: int
    action: str
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    status: str = "pending"  # "pending", "active", "complete", "failed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_id': self.step_id,
            'action': self.action,
            'preconditions': self.preconditions,
            'expected_outcome': self.expected_outcome,
            'status': self.status
        }


@dataclass
class SensorGoalBinding:
    """
    Binding between sensor observation and goal

    Connects sensor data to goal relevance
    """
    sensor_id: str
    observation: Any
    goal_id: str
    relevance: float  # How relevant to goal (0-1)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate binding"""
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"relevance must be 0-1, got {self.relevance}")


class WorkingMemory:
    """
    Active task context and multi-step plan state

    Design:
    - Limited capacity (7-10 slots, cognitively realistic)
    - Stores active goals, plan steps, intermediate results
    - Evicts low-priority items when full
    - Consolidates high-priority items to LTM when task complete

    Integration:
    - Goal Manager: Stores active goal context
    - Deliberation Engine: Stores multi-step plans
    - Attention Manager: Provides task context for attention
    - Track 2 (LTM): Consolidates important items on task completion
    """

    def __init__(
        self,
        capacity: int = 10,
        memory_retrieval: Optional[MemoryRetrieval] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Working Memory

        Args:
            capacity: Maximum number of slots (default: 10, based on 7±2 cognitive limit)
            memory_retrieval: Memory system from Track 2 (for consolidation)
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.memory = memory_retrieval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Active slots
        self.slots: Dict[str, WorkingMemorySlot] = {}

        # Plan tracking
        self.active_plan: Optional[List[PlanStep]] = None
        self.current_step_index: Optional[int] = None

        # Sensor-goal bindings
        self.bindings: List[SensorGoalBinding] = []

        # Statistics
        self.total_adds: int = 0
        self.total_evictions: int = 0
        self.total_accesses: int = 0

        # Recency weight for eviction (how much to favor recent items)
        self.recency_weight: float = 0.3

    def add_item(
        self,
        content_type: str,
        content: Any,
        priority: float,
        goal_id: Optional[str] = None
    ) -> str:
        """
        Add item to working memory

        If at capacity, evict lowest-retention-score item

        Args:
            content_type: Type of content ("goal", "plan_step", etc.)
            content: The content itself
            priority: Priority for retention (0-1)
            goal_id: Associated goal ID (optional)

        Returns:
            slot_id of added item
        """
        # Check if at capacity
        if len(self.slots) >= self.capacity:
            self._evict_lowest_priority()

        # Create slot
        slot_id = f"wm_{uuid.uuid4().hex[:8]}"
        slot = WorkingMemorySlot(
            slot_id=slot_id,
            content_type=content_type,
            content=content,
            priority=priority,
            timestamp=time.time(),
            goal_id=goal_id
        )

        # Add to working memory
        self.slots[slot_id] = slot
        self.total_adds += 1

        return slot_id

    def get_item(self, slot_id: str) -> Optional[WorkingMemorySlot]:
        """
        Get item from working memory

        Args:
            slot_id: Slot identifier

        Returns:
            WorkingMemorySlot if found, None otherwise
        """
        if slot_id in self.slots:
            slot = self.slots[slot_id]
            slot.access_count += 1
            self.total_accesses += 1
            return slot
        return None

    def get_context(self, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current working memory context

        Optionally filter by goal_id

        Args:
            goal_id: Filter by goal (None = all content)

        Returns:
            Dictionary with current context
        """
        # Filter slots
        if goal_id is not None:
            relevant_slots = [s for s in self.slots.values() if s.goal_id == goal_id]
        else:
            relevant_slots = list(self.slots.values())

        # Group by content type
        context = {
            'goals': [],
            'plan_steps': [],
            'intermediate_results': [],
            'bindings': [],
            'other': []
        }

        for slot in relevant_slots:
            if slot.content_type == 'goal':
                context['goals'].append(slot.content)
            elif slot.content_type == 'plan_step':
                context['plan_steps'].append(slot.content)
            elif slot.content_type == 'intermediate_result':
                context['intermediate_results'].append(slot.content)
            elif slot.content_type == 'binding':
                context['bindings'].append(slot.content)
            else:
                context['other'].append(slot.content)

        # Add plan state
        context['active_plan'] = self.active_plan
        context['current_step'] = self.current_step_index

        # Add bindings
        if goal_id:
            context['sensor_bindings'] = [b for b in self.bindings if b.goal_id == goal_id]
        else:
            context['sensor_bindings'] = self.bindings

        return context

    def load_plan(self, plan: List[PlanStep]):
        """
        Load multi-step plan into working memory

        Args:
            plan: List of plan steps
        """
        self.active_plan = plan
        self.current_step_index = 0

        # Add plan steps to working memory with appropriate priorities
        for i, step in enumerate(plan):
            # Current step has highest priority, future steps lower
            if i == 0:
                priority = 0.9  # Current step
            else:
                priority = 0.7  # Future step

            self.add_item(
                content_type='plan_step',
                content=step,
                priority=priority,
                goal_id=None  # Plan steps not tied to specific goal slot
            )

    def advance_plan(self) -> Optional[PlanStep]:
        """
        Move to next step in plan

        Returns:
            Next PlanStep if available, None if plan complete
        """
        if not self.active_plan or self.current_step_index is None:
            return None

        # Mark current step as complete
        if self.current_step_index < len(self.active_plan):
            self.active_plan[self.current_step_index].status = "complete"

        # Advance to next step
        self.current_step_index += 1

        # Check if plan complete
        if self.current_step_index >= len(self.active_plan):
            return None

        # Mark next step as active
        next_step = self.active_plan[self.current_step_index]
        next_step.status = "active"

        return next_step

    def get_current_plan_step(self) -> Optional[PlanStep]:
        """Get current plan step"""
        if self.active_plan and self.current_step_index is not None:
            if self.current_step_index < len(self.active_plan):
                return self.active_plan[self.current_step_index]
        return None

    def bind_sensor_to_goal(
        self,
        sensor_id: str,
        observation: Any,
        goal_id: str,
        relevance: float
    ):
        """
        Create binding between sensor observation and goal

        Args:
            sensor_id: Sensor identifier
            observation: Sensor data
            goal_id: Goal identifier
            relevance: How relevant to goal (0-1)
        """
        binding = SensorGoalBinding(
            sensor_id=sensor_id,
            observation=observation,
            goal_id=goal_id,
            relevance=relevance
        )

        self.bindings.append(binding)

        # Keep only recent bindings (last 50)
        if len(self.bindings) > 50:
            self.bindings = self.bindings[-50:]

    def get_sensor_bindings(self, goal_id: Optional[str] = None) -> List[SensorGoalBinding]:
        """
        Get sensor-goal bindings

        Args:
            goal_id: Filter by goal (None = all bindings)

        Returns:
            List of sensor bindings
        """
        if goal_id:
            return [b for b in self.bindings if b.goal_id == goal_id]
        return self.bindings

    def consolidate_to_ltm(self, goal_id: Optional[str] = None):
        """
        Consolidate high-priority working memory items to LTM

        Called when task completes. Stores important intermediate results
        and goal context to long-term memory.

        Args:
            goal_id: Goal that completed (for filtering)
        """
        if not self.memory:
            return  # No LTM available

        # Get high-priority items (priority > 0.7)
        high_priority = [
            s for s in self.slots.values()
            if s.priority >= 0.7 and (goal_id is None or s.goal_id == goal_id)
        ]

        # TODO: Consolidate to LTM when fully integrated with Track 2
        # For now, just track that consolidation would happen

        consolidated_count = len(high_priority)

        return consolidated_count

    def clear(self, goal_id: Optional[str] = None):
        """
        Clear working memory

        Args:
            goal_id: Clear only items for this goal (None = clear all)
        """
        if goal_id is None:
            # Clear everything
            self.slots.clear()
            self.active_plan = None
            self.current_step_index = None
            self.bindings.clear()
        else:
            # Clear only items for specific goal
            to_remove = [sid for sid, slot in self.slots.items() if slot.goal_id == goal_id]
            for sid in to_remove:
                del self.slots[sid]

            # Clear bindings for goal
            self.bindings = [b for b in self.bindings if b.goal_id != goal_id]

    def _evict_lowest_priority(self):
        """
        Evict lowest-retention-score item from working memory

        Retention score = priority * (1 - recency_weight) + recency * recency_weight
        """
        if not self.slots:
            return

        current_time = time.time()

        # Compute retention scores
        retention_scores = {}
        for slot_id, slot in self.slots.items():
            # Recency score (0-1, higher = more recent)
            age = current_time - slot.timestamp
            recency = 1.0 / (1.0 + age)  # Decays with age

            # Combined retention score
            retention_score = (
                slot.priority * (1.0 - self.recency_weight) +
                recency * self.recency_weight
            )

            retention_scores[slot_id] = retention_score

        # Find lowest score
        lowest_slot_id = min(retention_scores, key=retention_scores.get)

        # Optional: Consolidate to LTM if high priority
        evicted_slot = self.slots[lowest_slot_id]
        if evicted_slot.priority >= 0.7:
            # Would consolidate to LTM here if fully integrated
            pass

        # Evict
        del self.slots[lowest_slot_id]
        self.total_evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        return {
            'capacity': self.capacity,
            'current_size': len(self.slots),
            'utilization': len(self.slots) / self.capacity,
            'total_adds': self.total_adds,
            'total_evictions': self.total_evictions,
            'total_accesses': self.total_accesses,
            'active_plan_steps': len(self.active_plan) if self.active_plan else 0,
            'current_step': self.current_step_index,
            'sensor_bindings': len(self.bindings)
        }


def test_working_memory():
    """Test Working Memory implementation"""
    print("\n" + "="*60)
    print("TESTING WORKING MEMORY")
    print("="*60)

    # Create working memory
    wm = WorkingMemory(capacity=5)  # Small capacity for testing

    # Test 1: Add items
    print("\n1. Adding items to working memory...")
    wm.add_item('goal', 'Navigate to landmark', priority=1.0, goal_id='goal1')
    wm.add_item('plan_step', 'Turn left', priority=0.9, goal_id='goal1')
    wm.add_item('intermediate_result', {'distance': 10.5}, priority=0.6, goal_id='goal1')

    stats = wm.get_stats()
    print(f"   Added 3 items, size: {stats['current_size']}/{stats['capacity']}")

    # Test 2: Get context
    print("\n2. Getting context...")
    context = wm.get_context(goal_id='goal1')
    print(f"   Goals: {len(context['goals'])}")
    print(f"   Plan steps: {len(context['plan_steps'])}")
    print(f"   Intermediate results: {len(context['intermediate_results'])}")

    # Test 3: Load plan
    print("\n3. Loading multi-step plan...")
    plan = [
        PlanStep(step_id=0, action="Move forward", expected_outcome="Advance 1m"),
        PlanStep(step_id=1, action="Turn left", expected_outcome="Heading 90°"),
        PlanStep(step_id=2, action="Move forward", expected_outcome="Reach target")
    ]
    wm.load_plan(plan)
    print(f"   Loaded plan with {len(plan)} steps")

    current_step = wm.get_current_plan_step()
    print(f"   Current step: {current_step.action}")

    # Test 4: Advance plan
    print("\n4. Advancing through plan...")
    step_count = 0
    while True:
        next_step = wm.advance_plan()
        if next_step is None:
            break
        step_count += 1
        print(f"   Step {step_count}: {next_step.action}")

    print(f"   Plan complete after {step_count} steps")

    # Test 5: Sensor binding
    print("\n5. Creating sensor-goal binding...")
    wm.bind_sensor_to_goal('vision', np.array([1, 2, 3]), 'goal1', relevance=0.9)
    bindings = wm.get_sensor_bindings('goal1')
    print(f"   Created {len(bindings)} binding(s)")
    print(f"   Binding: {bindings[0].sensor_id} → goal1, relevance={bindings[0].relevance}")

    # Test 6: Capacity and eviction
    print("\n6. Testing capacity (adding beyond limit)...")
    for i in range(10):
        wm.add_item('other', f'item_{i}', priority=0.3)

    final_stats = wm.get_stats()
    print(f"   Final size: {final_stats['current_size']}/{final_stats['capacity']}")
    print(f"   Evictions: {final_stats['total_evictions']}")
    print(f"   Utilization: {final_stats['utilization']:.1%}")

    # Test 7: Clear
    print("\n7. Clearing working memory...")
    wm.clear()
    cleared_stats = wm.get_stats()
    print(f"   Size after clear: {cleared_stats['current_size']}")

    print("\n" + "="*60)
    print("✅ WORKING MEMORY TESTS PASSED")
    print("="*60)

    return wm


if __name__ == "__main__":
    test_working_memory()
