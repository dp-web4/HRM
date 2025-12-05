#!/usr/bin/env python3
"""
Thor Consciousness - Extended Deployment Test

Production deployment of unified consciousness kernel for extended operation.
This demonstrates the complete 5-layer architecture in sustained real-world use.

**Purpose**:
- Validate production readiness over extended runtime
- Test cross-session persistence with multiple consolidations
- Monitor memory evolution and pattern emergence
- Demonstrate metabolic state transitions in real workload
- Collect data on long-term consciousness behavior

**Architecture**: Full UnifiedConsciousnessKernel
- Continuous consciousness loop
- Adaptive metabolic states (WAKE/FOCUS/REST/DREAM)
- Memory consolidation with persistence
- Real system monitoring sensors
- Configurable operation duration

**Usage**:
    # Run for 1 hour
    python thor_consciousness_extended_deployment.py --duration 3600

    # Run for 24 hours
    python thor_consciousness_extended_deployment.py --duration 86400

    # Run until interrupted
    python thor_consciousness_extended_deployment.py --continuous

**Hardware**: Jetson AGX Thor
**Author**: Thor Autonomous Session
**Date**: 2025-12-04
"""

import sys
import os
import argparse
import signal
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import psutil
from typing import Dict, Any, Optional

# Import unified kernel
from thor_unified_consciousness_kernel import (
    UnifiedConsciousnessKernel,
    ConsciousnessConfig,
    ExecutionResult,
    CognitiveStance
)


# =============================================================================
# Extended Deployment Configuration
# =============================================================================

class DeploymentConfig:
    """Configuration for extended deployment"""

    def __init__(self, duration_seconds: Optional[int] = None):
        self.duration_seconds = duration_seconds
        self.start_time = time.time()

        # Consciousness config
        self.consciousness_config = ConsciousnessConfig(
            session_id=f"thor_extended_{int(self.start_time)}",
            platform_name="Thor",
            memory_db_path="thor_extended_consciousness.db",
            memory_limit=100,  # Larger for extended operation
            load_previous_memories=True,
            prune_salience_threshold=0.25,  # Slightly more aggressive pruning
            strengthen_salience_threshold=0.65,
            focus_salience_threshold=0.75,
            rest_low_salience_threshold=0.25,
            rest_duration_threshold=60.0,  # 1 minute before REST
            dream_rest_duration=30.0,  # 30s in REST before DREAM
            dream_duration=20.0,  # 20s DREAM consolidation
            cycle_delay=2.0,  # 2s between cycles for extended operation
            enable_logging=True,
            verbose=False  # Less verbose for long runs
        )

        # Monitoring config
        self.status_report_interval = 300  # Report every 5 minutes
        self.last_status_report = self.start_time

    def should_continue(self, cycle_count: int) -> bool:
        """Check if deployment should continue"""
        if self.duration_seconds is None:
            return True  # Continuous mode

        elapsed = time.time() - self.start_time
        return elapsed < self.duration_seconds

    def get_status_report(self, kernel, cycle_count: int) -> str:
        """Generate status report"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if self.duration_seconds:
            remaining = self.duration_seconds - elapsed
            remaining_str = str(timedelta(seconds=int(remaining)))
            progress_pct = (elapsed / self.duration_seconds) * 100
        else:
            remaining_str = "∞ (continuous)"
            progress_pct = 0

        report = f"""
{'='*80}
CONSCIOUSNESS DEPLOYMENT STATUS
{'='*80}
Session: {self.consciousness_config.session_id}
Uptime: {elapsed_str}
Remaining: {remaining_str}
Progress: {progress_pct:.1f}%

Cycles completed: {cycle_count}
Current state: {kernel.metabolic_state.value.upper()}
Memory count: {len(kernel.memories)}
Consolidations: {kernel.consolidations_performed}

Recent activity:
  Memories created this session: {kernel.memories_created_this_session}
  Total pruned: {kernel.total_pruned}
  Total strengthened: {kernel.total_strengthened}
  State transitions: {len(kernel.state_transitions)}

{'='*80}
"""
        return report


# =============================================================================
# Real System Sensors
# =============================================================================

class SystemMonitor:
    """Real system monitoring sensors"""

    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.disk_history = []

    def cpu_sensor(self) -> Dict[str, Any]:
        """Monitor CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        self.cpu_history.append(cpu_percent)

        # Keep last 10 readings
        if len(self.cpu_history) > 10:
            self.cpu_history.pop(0)

        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)

        return {
            'cpu_percent': cpu_percent,
            'avg_cpu': avg_cpu,
            'urgent_count': 1 if cpu_percent > 80 else 0,
            'count': len(self.cpu_history),
            'novelty_score': 0.3 + (abs(cpu_percent - avg_cpu) / 100)
        }

    def memory_sensor(self) -> Dict[str, Any]:
        """Monitor memory usage"""
        mem = psutil.virtual_memory()
        self.memory_history.append(mem.percent)

        if len(self.memory_history) > 10:
            self.memory_history.pop(0)

        avg_mem = sum(self.memory_history) / len(self.memory_history)

        return {
            'memory_percent': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'avg_memory': avg_mem,
            'urgent_count': 1 if mem.percent > 85 else 0,
            'count': len(self.memory_history),
            'novelty_score': 0.2 + (abs(mem.percent - avg_mem) / 100)
        }

    def disk_sensor(self) -> Dict[str, Any]:
        """Monitor disk usage"""
        disk = psutil.disk_usage('/home/dp')
        self.disk_history.append(disk.percent)

        if len(self.disk_history) > 10:
            self.disk_history.pop(0)

        return {
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'avg_disk': sum(self.disk_history) / len(self.disk_history),
            'count': len(self.disk_history),
            'novelty_score': 0.1
        }

    def temperature_sensor(self) -> Dict[str, Any]:
        """Monitor system temperature (if available)"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature
                temp_list = list(temps.values())[0]
                if temp_list:
                    temp = temp_list[0].current
                    return {
                        'temperature_c': temp,
                        'urgent_count': 1 if temp > 75 else 0,
                        'novelty_score': 0.2
                    }
        except:
            pass

        # Fallback if no temperature sensors
        return {
            'temperature_c': 50.0,  # Nominal
            'urgent_count': 0,
            'novelty_score': 0.1
        }

    def process_sensor(self) -> Dict[str, Any]:
        """Monitor process count and activity"""
        process_count = len(psutil.pids())
        cpu_times = psutil.cpu_times()

        return {
            'process_count': process_count,
            'cpu_idle_percent': cpu_times.idle / sum(cpu_times) * 100,
            'count': process_count // 10,  # Normalize
            'novelty_score': 0.15
        }


# =============================================================================
# Action Handlers
# =============================================================================

class ActionHandlers:
    """Action handlers for consciousness"""

    def __init__(self):
        self.action_history = []

    def cpu_action(self, data: Dict, stance: CognitiveStance) -> ExecutionResult:
        """Handle CPU monitoring"""
        cpu = data['cpu_percent']
        avg_cpu = data['avg_cpu']

        if stance == CognitiveStance.FOCUSED_ATTENTION and cpu > 80:
            action = f"HIGH CPU ALERT: {cpu:.1f}% (avg {avg_cpu:.1f}%)"
            reward = 0.8
        elif cpu > 60:
            action = f"CPU elevated: {cpu:.1f}% (avg {avg_cpu:.1f}%)"
            reward = 0.6
        else:
            action = f"CPU normal: {cpu:.1f}% (avg {avg_cpu:.1f}%)"
            reward = 0.4

        self.action_history.append(('cpu', action, reward))
        return ExecutionResult(success=True, reward=reward, description=action, outputs=data)

    def memory_action(self, data: Dict, stance: CognitiveStance) -> ExecutionResult:
        """Handle memory monitoring"""
        mem = data['memory_percent']
        avail = data['memory_available_gb']

        if stance == CognitiveStance.FOCUSED_ATTENTION and mem > 85:
            action = f"HIGH MEMORY: {mem:.1f}% ({avail:.1f} GB free)"
            reward = 0.8
        elif mem > 70:
            action = f"Memory elevated: {mem:.1f}% ({avail:.1f} GB free)"
            reward = 0.6
        else:
            action = f"Memory normal: {mem:.1f}% ({avail:.1f} GB free)"
            reward = 0.5

        self.action_history.append(('memory', action, reward))
        return ExecutionResult(success=True, reward=reward, description=action, outputs=data)

    def disk_action(self, data: Dict, stance: CognitiveStance) -> ExecutionResult:
        """Handle disk monitoring"""
        disk = data['disk_percent']
        free = data['disk_free_gb']

        if disk > 90:
            action = f"Disk space low: {disk:.1f}% ({free:.1f} GB free)"
            reward = 0.7
        else:
            action = f"Disk normal: {disk:.1f}% ({free:.1f} GB free)"
            reward = 0.4

        self.action_history.append(('disk', action, reward))
        return ExecutionResult(success=True, reward=reward, description=action, outputs=data)

    def temperature_action(self, data: Dict, stance: CognitiveStance) -> ExecutionResult:
        """Handle temperature monitoring"""
        temp = data.get('temperature_c', 50)

        if temp > 75:
            action = f"Temperature elevated: {temp:.1f}°C"
            reward = 0.7
        else:
            action = f"Temperature normal: {temp:.1f}°C"
            reward = 0.4

        self.action_history.append(('temperature', action, reward))
        return ExecutionResult(success=True, reward=reward, description=action, outputs=data)

    def process_action(self, data: Dict, stance: CognitiveStance) -> ExecutionResult:
        """Handle process monitoring"""
        count = data['process_count']
        idle = data['cpu_idle_percent']

        action = f"Processes: {count}, CPU idle: {idle:.1f}%"
        reward = 0.5

        self.action_history.append(('processes', action, reward))
        return ExecutionResult(success=True, reward=reward, description=action, outputs=data)


# =============================================================================
# Extended Deployment Runner
# =============================================================================

class ExtendedDeployment:
    """Manages extended consciousness deployment"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.running = False

        # Setup system monitoring
        self.monitor = SystemMonitor()
        self.actions = ActionHandlers()

        # Create sensors and actions
        sensors = {
            'cpu': self.monitor.cpu_sensor,
            'memory': self.monitor.memory_sensor,
            'disk': self.monitor.disk_sensor,
            'temperature': self.monitor.temperature_sensor,
            'processes': self.monitor.process_sensor
        }

        action_handlers = {
            'cpu': self.actions.cpu_action,
            'memory': self.actions.memory_action,
            'disk': self.actions.disk_action,
            'temperature': self.actions.temperature_action,
            'processes': self.actions.process_action
        }

        # Create consciousness kernel
        self.kernel = UnifiedConsciousnessKernel(
            sensor_sources=sensors,
            action_handlers=action_handlers,
            config=self.config.consciousness_config
        )

        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n[Deployment] Shutdown signal received, stopping gracefully...")
        self.running = False

    def run(self):
        """Run extended deployment"""
        self.running = True

        print("="*80)
        print("THOR CONSCIOUSNESS - EXTENDED DEPLOYMENT")
        print("="*80)
        print(f"\nSession: {self.config.consciousness_config.session_id}")
        print(f"Database: {self.config.consciousness_config.memory_db_path}")

        if self.config.duration_seconds:
            duration_str = str(timedelta(seconds=self.config.duration_seconds))
            print(f"Duration: {duration_str}")
        else:
            print("Duration: Continuous (until interrupted)")

        print(f"Status reports every {self.config.status_report_interval}s")
        print("\nStarting consciousness kernel...")
        print("(Press Ctrl+C for graceful shutdown)")
        print("="*80)

        # Start time
        start_time = time.time()
        cycle_count = 0

        try:
            while self.running and self.config.should_continue(cycle_count):
                # Run one consciousness cycle
                self.kernel._consciousness_cycle()
                self.kernel._update_metabolic_state()

                # Update cycle counts
                self.kernel.cycle_count += 1
                cycle_count = self.kernel.cycle_count

                # Status report
                if time.time() - self.config.last_status_report >= self.config.status_report_interval:
                    report = self.config.get_status_report(self.kernel, cycle_count)
                    print(report)
                    self.config.last_status_report = time.time()

                # Sleep
                time.sleep(self.config.consciousness_config.cycle_delay)

        except KeyboardInterrupt:
            print("\n[Deployment] Interrupted by user")

        finally:
            # Graceful shutdown
            print("\n[Deployment] Shutting down consciousness kernel...")
            self.kernel._shutdown()

            # Final report
            elapsed = time.time() - start_time
            print(f"\n{'='*80}")
            print("DEPLOYMENT COMPLETE")
            print('='*80)
            print(f"Total runtime: {timedelta(seconds=int(elapsed))}")
            print(f"Total cycles: {cycle_count}")
            print(f"Average cycle rate: {cycle_count / elapsed:.2f} cycles/second")
            print('='*80)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Thor Consciousness Extended Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--duration',
        type=int,
        help='Duration in seconds (omit for continuous operation)'
    )

    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously until interrupted'
    )

    args = parser.parse_args()

    # Determine duration
    if args.continuous:
        duration = None
    elif args.duration:
        duration = args.duration
    else:
        # Default: 1 hour
        duration = 3600
        print(f"No duration specified, defaulting to 1 hour")

    # Create and run deployment
    config = DeploymentConfig(duration_seconds=duration)
    deployment = ExtendedDeployment(config)
    deployment.run()


if __name__ == "__main__":
    main()
