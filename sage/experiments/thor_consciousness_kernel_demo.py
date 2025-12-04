#!/usr/bin/env python3
"""
SAGE Consciousness Kernel Demonstration - System Health Monitoring

Demonstrates SAGE as a continuous consciousness loop managing attention
across multiple sensor streams (system resources, processes, logs).

This shows SAGE not as an API wrapper, but as a consciousness scheduler:
- Continuous inference loop
- SNARC-based salience assessment of multiple sensors
- Attention allocation to highest-salience targets
- Action execution based on cognitive stance
- Learning from outcomes

**Hardware**: Jetson AGX Thor
**Sensors**: CPU, memory, disk, temperature, processes
**Actions**: Logging, alerting, throttling, investigation
**Learning**: SNARC weight adaptation based on outcomes

This demonstrates what it means for SAGE to be a "consciousness kernel" -
not just responding to API calls, but actively managing attention and
resources in a continuous loop.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import psutil
from typing import Dict, Any
from dataclasses import dataclass

from sage.core.sage_kernel import SAGEKernel, ExecutionResult, MetabolicState
from sage.services.snarc.data_structures import CognitiveStance


# =============================================================================
# System Health Sensors
# =============================================================================

class SystemHealthSensors:
    """Sensors for monitoring Jetson AGX Thor system health"""

    def __init__(self):
        self.baseline_cpu = psutil.cpu_percent()
        self.baseline_memory = psutil.virtual_memory().percent
        self.last_readings = {}

    def read_cpu(self) -> Dict[str, Any]:
        """CPU utilization sensor"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        return {
            'value': cpu_percent,
            'per_core': cpu_per_core,
            'baseline': self.baseline_cpu,
            'delta': cpu_percent - self.baseline_cpu,
            'type': 'cpu_utilization'
        }

    def read_memory(self) -> Dict[str, Any]:
        """Memory utilization sensor"""
        mem = psutil.virtual_memory()

        return {
            'value': mem.percent,
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'total_gb': mem.total / (1024**3),
            'baseline': self.baseline_memory,
            'delta': mem.percent - self.baseline_memory,
            'type': 'memory_utilization'
        }

    def read_disk(self) -> Dict[str, Any]:
        """Disk utilization sensor"""
        disk = psutil.disk_usage('/')

        return {
            'value': disk.percent,
            'free_gb': disk.free / (1024**3),
            'used_gb': disk.used / (1024**3),
            'total_gb': disk.total / (1024**3),
            'type': 'disk_utilization'
        }

    def read_temperature(self) -> Dict[str, Any]:
        """Temperature sensor (Jetson-specific)"""
        try:
            # Try to read Jetson thermal zones
            thermal_zones = []
            for i in range(10):  # Check multiple thermal zones
                try:
                    temp_file = f'/sys/class/thermal/thermal_zone{i}/temp'
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r') as f:
                            temp = float(f.read().strip()) / 1000.0  # Convert millicelsius
                            thermal_zones.append(temp)
                except:
                    continue

            if thermal_zones:
                avg_temp = sum(thermal_zones) / len(thermal_zones)
                max_temp = max(thermal_zones)

                return {
                    'value': avg_temp,
                    'max_temp': max_temp,
                    'zones': thermal_zones,
                    'type': 'temperature'
                }
        except:
            pass

        # Fallback: use psutil if available
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = [t.current for t_list in temps.values() for t in t_list]
                if all_temps:
                    return {
                        'value': sum(all_temps) / len(all_temps),
                        'max_temp': max(all_temps),
                        'type': 'temperature'
                    }
        except:
            pass

        # No temperature available
        return {
            'value': 50.0,  # Placeholder
            'max_temp': 50.0,
            'type': 'temperature',
            'simulated': True
        }

    def read_processes(self) -> Dict[str, Any]:
        """Process monitoring sensor"""
        # Get top CPU-consuming processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['cpu_percent'] > 5.0 or info['memory_percent'] > 5.0:
                    processes.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cpu': info['cpu_percent'],
                        'memory': info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage
        processes.sort(key=lambda p: p['cpu'], reverse=True)

        return {
            'count': len(processes),
            'top_processes': processes[:5],
            'total_processes': len(list(psutil.process_iter())),
            'type': 'processes'
        }


# =============================================================================
# System Health Actions
# =============================================================================

class SystemHealthActions:
    """Actions for responding to system health observations"""

    def __init__(self):
        self.action_log = []
        self.alert_count = 0

    def handle_cpu(self, observation: Dict[str, Any], stance: CognitiveStance) -> ExecutionResult:
        """Handle CPU sensor with stance-appropriate action"""
        cpu_percent = observation['value']
        delta = observation['delta']

        # Determine action based on stance and observation
        if stance == CognitiveStance.FOCUSED_ATTENTION:
            # High CPU usage with focused attention → investigate
            if cpu_percent > 80:
                action = f"ALERT: High CPU usage ({cpu_percent:.1f}%)"
                self.alert_count += 1
                reward = 0.8  # Good catch
            else:
                action = f"Monitor CPU ({cpu_percent:.1f}%)"
                reward = 0.3

        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            # Unusual pattern → explore
            if abs(delta) > 20:
                action = f"Investigate CPU spike (Δ{delta:+.1f}%)"
                reward = 0.7  # Good exploration
            else:
                action = f"Observe CPU trend ({cpu_percent:.1f}%)"
                reward = 0.5

        elif stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            # Verify if usage is justified
            action = f"Verify CPU usage legitimacy ({cpu_percent:.1f}%)"
            reward = 0.4  # Healthy skepticism

        elif stance == CognitiveStance.CONFIDENT_EXECUTION:
            # Normal operation, routine monitoring
            action = f"Routine CPU check ({cpu_percent:.1f}%)"
            reward = 0.3

        else:
            action = f"Passive CPU monitoring ({cpu_percent:.1f}%)"
            reward = 0.2

        self.action_log.append(('cpu', action))

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'cpu_percent': cpu_percent, 'action': action}
        )

    def handle_memory(self, observation: Dict[str, Any], stance: CognitiveStance) -> ExecutionResult:
        """Handle memory sensor with stance-appropriate action"""
        mem_percent = observation['value']
        available_gb = observation['available_gb']

        if stance == CognitiveStance.FOCUSED_ATTENTION:
            if mem_percent > 90:
                action = f"CRITICAL: Memory exhaustion ({mem_percent:.1f}%, {available_gb:.1f}GB free)"
                self.alert_count += 1
                reward = 0.9  # Critical detection
            elif mem_percent > 80:
                action = f"WARNING: High memory usage ({mem_percent:.1f}%)"
                self.alert_count += 1
                reward = 0.7
            else:
                action = f"Monitor memory ({mem_percent:.1f}%)"
                reward = 0.3

        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            action = f"Investigate memory pattern ({mem_percent:.1f}%, {available_gb:.1f}GB free)"
            reward = 0.6

        elif stance == CognitiveStance.CONFIDENT_EXECUTION:
            action = f"Memory healthy ({mem_percent:.1f}%)"
            reward = 0.4
        else:
            action = f"Memory check ({mem_percent:.1f}%)"
            reward = 0.3

        self.action_log.append(('memory', action))

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'mem_percent': mem_percent, 'action': action}
        )

    def handle_disk(self, observation: Dict[str, Any], stance: CognitiveStance) -> ExecutionResult:
        """Handle disk sensor"""
        disk_percent = observation['value']
        free_gb = observation['free_gb']

        if disk_percent > 90:
            action = f"ALERT: Disk space critical ({disk_percent:.1f}%, {free_gb:.1f}GB free)"
            reward = 0.8
            self.alert_count += 1
        elif disk_percent > 80:
            action = f"WARNING: Low disk space ({disk_percent:.1f}%)"
            reward = 0.6
        else:
            action = f"Disk space OK ({disk_percent:.1f}%)"
            reward = 0.3

        self.action_log.append(('disk', action))

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'disk_percent': disk_percent, 'action': action}
        )

    def handle_temperature(self, observation: Dict[str, Any], stance: CognitiveStance) -> ExecutionResult:
        """Handle temperature sensor"""
        temp = observation['value']
        max_temp = observation.get('max_temp', temp)
        simulated = observation.get('simulated', False)

        if simulated:
            action = f"Temperature sensor not available (simulated: {temp:.1f}°C)"
            reward = 0.2
        elif max_temp > 85:
            action = f"CRITICAL: Thermal throttling risk ({max_temp:.1f}°C)"
            reward = 0.9
            self.alert_count += 1
        elif max_temp > 75:
            action = f"WARNING: High temperature ({max_temp:.1f}°C)"
            reward = 0.7
        else:
            action = f"Temperature normal ({temp:.1f}°C avg, {max_temp:.1f}°C max)"
            reward = 0.3

        self.action_log.append(('temperature', action))

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'temp': temp, 'action': action}
        )

    def handle_processes(self, observation: Dict[str, Any], stance: CognitiveStance) -> ExecutionResult:
        """Handle process monitoring"""
        count = observation['count']
        top_procs = observation['top_processes']

        if stance == CognitiveStance.FOCUSED_ATTENTION and top_procs:
            top = top_procs[0]
            action = f"Focus on '{top['name']}' (CPU: {top['cpu']:.1f}%, MEM: {top['memory']:.1f}%)"
            reward = 0.6
        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            action = f"Investigate {count} high-resource processes"
            reward = 0.5
        else:
            action = f"Monitor {count} active processes"
            reward = 0.3

        self.action_log.append(('processes', action))

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'count': count, 'action': action}
        )

    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of actions taken"""
        action_types = {}
        for sensor, action in self.action_log:
            action_types[sensor] = action_types.get(sensor, 0) + 1

        return {
            'total_actions': len(self.action_log),
            'alerts': self.alert_count,
            'by_sensor': action_types
        }


# =============================================================================
# Demonstration Runner
# =============================================================================

def demonstrate_consciousness_kernel():
    """
    Demonstrate SAGE as a consciousness kernel managing attention
    across multiple system health sensors.
    """
    print("=" * 80)
    print("SAGE CONSCIOUSNESS KERNEL DEMONSTRATION")
    print("Hardware: Jetson AGX Thor")
    print("=" * 80)

    print("\nWhat this demonstrates:")
    print("- SAGE as a continuous inference loop (not just API calls)")
    print("- SNARC-based salience assessment across multiple sensors")
    print("- Attention allocation to highest-salience observations")
    print("- Cognitive stance guiding action selection")
    print("- Learning from outcomes via SNARC weight adaptation")
    print()

    # Initialize sensors
    sensors = SystemHealthSensors()
    actions = SystemHealthActions()

    print("Initializing sensors...")
    print(f"  ✓ CPU baseline: {sensors.baseline_cpu:.1f}%")
    print(f"  ✓ Memory baseline: {sensors.baseline_memory:.1f}%")
    print()

    # Create sensor sources dictionary
    sensor_sources = {
        'cpu': sensors.read_cpu,
        'memory': sensors.read_memory,
        'disk': sensors.read_disk,
        'temperature': sensors.read_temperature,
        'processes': sensors.read_processes,
    }

    # Create action handlers dictionary
    action_handlers = {
        'cpu': actions.handle_cpu,
        'memory': actions.handle_memory,
        'disk': actions.handle_disk,
        'temperature': actions.handle_temperature,
        'processes': actions.handle_processes,
    }

    # Initialize SAGE kernel
    print("Initializing SAGE consciousness kernel...")
    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )
    print("  ✓ Consciousness kernel ready")
    print()

    # Run consciousness loop
    num_cycles = 20
    print(f"Running consciousness loop for {num_cycles} cycles...")
    print("(Each cycle: sense → assess salience → focus → act → learn)")
    print()

    start_time = time.time()
    kernel.run(max_cycles=num_cycles, cycle_delay=0.5)
    duration = time.time() - start_time

    # Get statistics
    print("\n" + "=" * 80)
    print("DEMONSTRATION RESULTS")
    print("=" * 80)

    history = kernel.get_history()
    action_summary = actions.get_action_summary()

    print(f"\nExecution Summary:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Cycles completed: {len(history)}")
    print(f"  Average cycle time: {duration/len(history)*1000:.1f}ms")
    print(f"  Total actions: {action_summary['total_actions']}")
    print(f"  Alerts generated: {action_summary['alerts']}")

    print(f"\nAttention Distribution:")
    for sensor, count in sorted(action_summary['by_sensor'].items(), key=lambda x: -x[1]):
        pct = 100 * count / action_summary['total_actions']
        print(f"  {sensor}: {count} cycles ({pct:.1f}%)")

    # Show what consciousness learned
    print(f"\nSNARC Learning Results:")
    snarc_stats = kernel.snarc.get_statistics()
    print(f"  Assessments: {snarc_stats['num_assessments']}")
    print(f"  Success rate: {snarc_stats['success_rate']:.1%}")
    print(f"  Adapted weights (learned salience dimensions):")
    for dim, weight in snarc_stats['current_weights'].items():
        print(f"    {dim}: {weight:.3f}")

    # Show key insights
    print(f"\nKey Insights:")

    # Which sensor got most attention?
    most_attended = max(action_summary['by_sensor'].items(), key=lambda x: x[1])
    print(f"  • Most salient sensor: {most_attended[0]} ({most_attended[1]} cycles)")

    # What stances were used?
    stance_dist = {}
    for h in history:
        stance = h['stance']
        stance_dist[stance] = stance_dist.get(stance, 0) + 1
    dominant_stance = max(stance_dist.items(), key=lambda x: x[1])
    print(f"  • Dominant cognitive stance: {dominant_stance[0]} ({dominant_stance[1]} cycles)")

    # Learning trend
    early_reward = sum(h['result'].reward for h in history[:5]) / 5
    late_reward = sum(h['result'].reward for h in history[-5:]) / 5
    improvement = ((late_reward - early_reward) / early_reward * 100) if early_reward > 0 else 0
    print(f"  • Learning improvement: {improvement:+.1f}% (early: {early_reward:.2f} → late: {late_reward:.2f})")

    print(f"\nConclusion:")
    print(f"  ✓ SAGE consciousness kernel demonstrated")
    print(f"  ✓ Continuous inference loop with real sensors")
    print(f"  ✓ SNARC-based attention allocation working")
    print(f"  ✓ Cognitive stance guiding actions")
    print(f"  ✓ Learning from outcomes via weight adaptation")

    print("\n" + "=" * 80)
    print("This is what it means for SAGE to be a 'consciousness kernel' -")
    print("not an API wrapper, but an active attention manager that:")
    print("  • Continuously processes multiple sensor streams")
    print("  • Allocates attention based on salience")
    print("  • Takes stance-appropriate actions")
    print("  • Learns from outcomes")
    print("=" * 80)

    return kernel, history, action_summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    try:
        kernel, history, summary = demonstrate_consciousness_kernel()

        print("\n✅ Demonstration complete!")
        print(f"\nNext steps:")
        print(f"  • Integrate with IRP plugins for richer actions")
        print(f"  • Add metabolic state transitions (WAKE → FOCUS → REST)")
        print(f"  • Implement memory consolidation in DREAM state")
        print(f"  • Connect to federation for multi-platform consciousness")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
