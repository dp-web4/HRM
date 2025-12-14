#!/usr/bin/env python3
"""
Real-Time Consciousness Monitoring - Session 46

Provides lightweight, real-time observation of consciousness system behavior
without interfering with operation.

Features:
- Live consciousness cycle tracking
- Metabolic state transition visualization
- Quality score trends
- Epistemic state distribution
- ATP allocation monitoring
- Terminal-based display (edge-compatible)

Design Principles:
- Read-only observation (no state modification)
- Minimal overhead (< 5% processing time)
- Real-time updates
- No GUI dependencies

Integration:
- Session 41: Unified Consciousness (observes ConsciousnessCycle)
- Session 40: Metabolic States (tracks ATP and state transitions)
- Session 27-29: Quality Metrics (monitors quality trends)
- Session 30-31: Epistemic States (tracks epistemic distribution)

Author: Thor (Autonomous Session 46)
Date: 2025-12-13
"""

import time
import sys
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime

# Terminal colors for better visualization
class Colors:
    """ANSI color codes for terminal display"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # States
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'

    # Metabolic states
    WAKE = '\033[92m'      # Green
    FOCUS = '\033[93m'     # Yellow
    REST = '\033[94m'      # Blue
    DREAM = '\033[95m'     # Magenta
    CRISIS = '\033[91m'    # Red


@dataclass
class CycleSnapshot:
    """
    Snapshot of consciousness cycle for monitoring.

    Lightweight extract from ConsciousnessCycle for observation.
    """
    cycle_number: int
    timestamp: float
    quality_score: float
    epistemic_state: str
    metabolic_state: str
    total_atp: float
    quality_atp: float
    epistemic_atp: float
    processing_time: float
    errors: int


class StateHistory:
    """
    Rolling window history of consciousness states.

    Maintains recent cycles for trend analysis and visualization.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize state history.

        Args:
            max_size: Maximum number of cycles to retain
        """
        self.max_size = max_size
        self.cycles: deque = deque(maxlen=max_size)
        self.metabolic_transitions: deque = deque(maxlen=50)
        self.quality_trend: deque = deque(maxlen=max_size)
        self.epistemic_distribution: Dict[str, int] = {}

    def add_cycle(self, snapshot: CycleSnapshot):
        """Add new cycle snapshot to history"""
        self.cycles.append(snapshot)

        # Track quality trend
        self.quality_trend.append(snapshot.quality_score)

        # Track epistemic distribution
        state = snapshot.epistemic_state
        self.epistemic_distribution[state] = self.epistemic_distribution.get(state, 0) + 1

    def add_metabolic_transition(self, from_state: str, to_state: str, trigger: str):
        """Record metabolic state transition"""
        self.metabolic_transitions.append({
            'timestamp': time.time(),
            'from': from_state,
            'to': to_state,
            'trigger': trigger
        })

    def get_recent(self, n: int = 10) -> List[CycleSnapshot]:
        """Get n most recent cycles"""
        return list(self.cycles)[-n:] if len(self.cycles) >= n else list(self.cycles)

    def get_quality_stats(self) -> Dict:
        """Get quality score statistics"""
        if not self.quality_trend:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0}

        scores = list(self.quality_trend)
        return {
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'current': scores[-1] if scores else 0.0
        }


class LiveDisplay:
    """
    Terminal-based live display of consciousness state.

    Provides real-time visualization of consciousness metrics.
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize live display.

        Args:
            update_interval: Seconds between display updates
        """
        self.update_interval = update_interval
        self.last_update = 0.0

    def clear_screen(self):
        """Clear terminal screen"""
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()

    def render_dashboard(self, history: StateHistory, current_snapshot: Optional[CycleSnapshot] = None):
        """
        Render complete consciousness dashboard.

        Args:
            history: State history to display
            current_snapshot: Most recent cycle snapshot
        """
        now = time.time()
        if now - self.last_update < self.update_interval:
            return

        self.last_update = now
        self.clear_screen()

        # Header
        print(f"{Colors.BOLD}{'='*80}")
        print(f"SAGE Consciousness Monitor - Thor")
        print(f"{'='*80}{Colors.RESET}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Current State
        if current_snapshot:
            self._render_current_state(current_snapshot)

        # Recent Cycles
        self._render_recent_cycles(history)

        # Quality Trends
        self._render_quality_trends(history)

        # Epistemic Distribution
        self._render_epistemic_distribution(history)

        # Metabolic Transitions
        self._render_metabolic_transitions(history)

        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

    def _render_current_state(self, snapshot: CycleSnapshot):
        """Render current consciousness state"""
        print(f"{Colors.BOLD}Current State (Cycle #{snapshot.cycle_number}){Colors.RESET}")
        print(f"  Metabolic: {self._colorize_metabolic(snapshot.metabolic_state)}")
        print(f"  Epistemic: {self._colorize_epistemic(snapshot.epistemic_state)}")
        print(f"  Quality: {self._render_quality_bar(snapshot.quality_score)} ({snapshot.quality_score:.3f})")
        print(f"  ATP: {self._render_atp_bar(snapshot.total_atp, 100)} ({snapshot.total_atp:.1f})")
        print(f"    - Quality ATP: {snapshot.quality_atp:.1f}")
        print(f"    - Epistemic ATP: {snapshot.epistemic_atp:.1f}")
        print(f"  Processing: {snapshot.processing_time*1000:.1f}ms")
        print()

    def _render_recent_cycles(self, history: StateHistory):
        """Render recent cycle history"""
        recent = history.get_recent(5)
        if not recent:
            return

        print(f"{Colors.BOLD}Recent Cycles{Colors.RESET}")
        for cycle in reversed(recent):
            quality_bar = self._render_mini_quality_bar(cycle.quality_score)
            metabolic = self._colorize_metabolic(cycle.metabolic_state[:4])
            epistemic = cycle.epistemic_state[:8].ljust(8)
            print(f"  #{cycle.cycle_number:3d} | {quality_bar} | {metabolic} | {epistemic} | {cycle.processing_time*1000:5.1f}ms")
        print()

    def _render_quality_trends(self, history: StateHistory):
        """Render quality score trends"""
        stats = history.get_quality_stats()
        if stats['mean'] == 0.0:
            return

        print(f"{Colors.BOLD}Quality Metrics{Colors.RESET}")
        print(f"  Current: {stats['current']:.3f}")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print()

    def _render_epistemic_distribution(self, history: StateHistory):
        """Render epistemic state distribution"""
        dist = history.epistemic_distribution
        if not dist:
            return

        total = sum(dist.values())
        print(f"{Colors.BOLD}Epistemic States (Total: {total}){Colors.RESET}")

        # Sort by frequency
        for state, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100
            bar = self._render_mini_bar(pct, 20)
            print(f"  {state.ljust(12)}: {bar} {count:3d} ({pct:5.1f}%)")
        print()

    def _render_metabolic_transitions(self, history: StateHistory):
        """Render recent metabolic state transitions"""
        transitions = list(history.metabolic_transitions)[-5:]
        if not transitions:
            return

        print(f"{Colors.BOLD}Recent Metabolic Transitions{Colors.RESET}")
        for trans in reversed(transitions):
            from_state = self._colorize_metabolic(trans['from'])
            to_state = self._colorize_metabolic(trans['to'])
            trigger = trans['trigger'][:30]
            print(f"  {from_state} → {to_state}  ({trigger})")
        print()

    def _colorize_metabolic(self, state: str) -> str:
        """Add color to metabolic state"""
        state_upper = state.upper()
        if 'WAKE' in state_upper:
            return f"{Colors.WAKE}{state}{Colors.RESET}"
        elif 'FOCUS' in state_upper:
            return f"{Colors.FOCUS}{state}{Colors.RESET}"
        elif 'REST' in state_upper:
            return f"{Colors.REST}{state}{Colors.RESET}"
        elif 'DREAM' in state_upper:
            return f"{Colors.DREAM}{state}{Colors.RESET}"
        elif 'CRISIS' in state_upper:
            return f"{Colors.CRISIS}{state}{Colors.RESET}"
        return state

    def _colorize_epistemic(self, state: str) -> str:
        """Add color to epistemic state"""
        state_upper = state.upper()
        if 'CONFIDENT' in state_upper:
            return f"{Colors.GREEN}{state}{Colors.RESET}"
        elif 'STABLE' in state_upper:
            return f"{Colors.CYAN}{state}{Colors.RESET}"
        elif 'LEARNING' in state_upper:
            return f"{Colors.YELLOW}{state}{Colors.RESET}"
        elif 'CONFUSED' in state_upper:
            return f"{Colors.MAGENTA}{state}{Colors.RESET}"
        elif 'FRUSTRATED' in state_upper:
            return f"{Colors.RED}{state}{Colors.RESET}"
        return state

    def _render_quality_bar(self, score: float, width: int = 30) -> str:
        """Render quality score as progress bar"""
        filled = int(score * width)
        bar = '█' * filled + '░' * (width - filled)
        if score >= 0.85:
            color = Colors.GREEN
        elif score >= 0.70:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        return f"{color}{bar}{Colors.RESET}"

    def _render_mini_quality_bar(self, score: float, width: int = 10) -> str:
        """Render mini quality score bar"""
        filled = int(score * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar

    def _render_atp_bar(self, current: float, max_val: float, width: int = 30) -> str:
        """Render ATP budget as progress bar"""
        filled = int((current / max_val) * width)
        bar = '█' * filled + '░' * (width - filled)
        if current >= 100:
            color = Colors.GREEN
        elif current >= 80:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        return f"{color}{bar}{Colors.RESET}"

    def _render_mini_bar(self, percentage: float, width: int = 20) -> str:
        """Render mini percentage bar"""
        filled = int((percentage / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar


class ConsciousnessMonitor:
    """
    Real-time consciousness system monitor.

    Observes UnifiedConsciousnessManager without interfering with operation.
    Provides live visualization of consciousness state and metrics.
    """

    def __init__(self,
                 history_size: int = 100,
                 display_interval: float = 1.0,
                 enabled: bool = True,
                 display_enabled: bool = True):
        """
        Initialize consciousness monitor.

        Args:
            history_size: Number of cycles to retain in history
            display_interval: Seconds between display updates
            enabled: Whether monitoring is active
            display_enabled: Whether to show live display (set False for testing)
        """
        self.history = StateHistory(max_size=history_size)
        self.display = LiveDisplay(update_interval=display_interval)
        self.enabled = enabled
        self.display_enabled = display_enabled
        self.monitoring_start = time.time()
        self.total_cycles_observed = 0
        self.total_overhead_time = 0.0

    def observe_cycle(self, cycle):
        """
        Observe a consciousness cycle.

        Args:
            cycle: ConsciousnessCycle from UnifiedConsciousnessManager
        """
        if not self.enabled:
            return

        observe_start = time.time()

        # Create snapshot
        snapshot = CycleSnapshot(
            cycle_number=cycle.cycle_number,
            timestamp=cycle.timestamp,
            quality_score=cycle.quality_score.normalized if cycle.quality_score else 0.0,
            epistemic_state=cycle.epistemic_state.value if cycle.epistemic_state else "unknown",
            metabolic_state=cycle.metabolic_state.value if hasattr(cycle.metabolic_state, 'value') else str(cycle.metabolic_state),
            total_atp=cycle.total_atp,
            quality_atp=cycle.quality_atp,
            epistemic_atp=cycle.epistemic_atp,
            processing_time=cycle.processing_time,
            errors=len(cycle.errors)
        )

        # Add to history
        self.history.add_cycle(snapshot)
        self.total_cycles_observed += 1

        # Update display (only if enabled)
        if self.display_enabled:
            self.display.render_dashboard(self.history, snapshot)

        # Track overhead
        observe_time = time.time() - observe_start
        self.total_overhead_time += observe_time

    def observe_metabolic_transition(self, from_state, to_state, trigger: str):
        """
        Observe metabolic state transition.

        Args:
            from_state: Previous metabolic state
            to_state: New metabolic state
            trigger: What triggered the transition
        """
        if not self.enabled:
            return

        from_str = from_state.value if hasattr(from_state, 'value') else str(from_state)
        to_str = to_state.value if hasattr(to_state, 'value') else str(to_state)

        self.history.add_metabolic_transition(from_str, to_str, trigger)

    def get_overhead_percentage(self) -> float:
        """Calculate monitoring overhead as percentage of total time"""
        if self.total_cycles_observed == 0:
            return 0.0

        total_runtime = time.time() - self.monitoring_start
        return (self.total_overhead_time / total_runtime) * 100 if total_runtime > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get comprehensive monitoring statistics"""
        return {
            'cycles_observed': self.total_cycles_observed,
            'monitoring_time': time.time() - self.monitoring_start,
            'overhead_percentage': self.get_overhead_percentage(),
            'quality_stats': self.history.get_quality_stats(),
            'epistemic_distribution': self.history.epistemic_distribution,
            'recent_transitions': len(self.history.metabolic_transitions)
        }


def example_usage():
    """Example demonstrating consciousness monitor"""
    import sys
    sys.path.insert(0, '/home/dp/ai-workspace/HRM')

    from sage.core.unified_consciousness import UnifiedConsciousnessManager

    print("Consciousness Monitor Demo")
    print("=" * 60)
    print()

    # Initialize consciousness and monitor
    consciousness = UnifiedConsciousnessManager()
    monitor = ConsciousnessMonitor(history_size=100, display_interval=0.5)

    # Simulate consciousness cycles
    test_prompts = [
        ("What is machine learning?",
         "Machine learning is a branch of artificial intelligence that uses algorithms like neural networks, decision trees, and support vector machines to learn from data."),
        ("Explain consciousness",
         "Consciousness involves awareness and subjective experience."),
        ("What is 2+2?",
         "2+2 equals 4."),
    ]

    try:
        for i, (prompt, response) in enumerate(test_prompts):
            # Run consciousness cycle
            cycle = consciousness.consciousness_cycle(
                prompt=prompt,
                response=response,
                task_salience=0.5 + (i * 0.2)
            )

            # Observe with monitor
            monitor.observe_cycle(cycle)

            # Simulate delay between cycles
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Show final statistics
    print("\n\nMonitoring Statistics:")
    stats = monitor.get_statistics()
    print(f"  Cycles observed: {stats['cycles_observed']}")
    print(f"  Monitoring overhead: {stats['overhead_percentage']:.2f}%")
    print(f"  Quality stats: {stats['quality_stats']}")


if __name__ == '__main__':
    example_usage()
