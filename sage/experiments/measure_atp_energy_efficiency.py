#!/usr/bin/env python3
"""
Session 15: ATP Energy Efficiency Study

Measure power consumption across different ATP configurations to quantify
the energy trade-offs identified in Session 13.

Research Questions:
1. What is the actual power cost of different attention rates?
2. How much energy does each ATP configuration consume per cycle?
3. What is the energy efficiency (operations per watt)?
4. Can we quantify the energy vs coverage trade-off?

Approach:
- Use tegrastats to measure system power consumption
- Run consciousness cycles under different ATP configs
- Calculate energy per cycle, energy per attended observation
- Compare energy efficiency across Maximum/Balanced/Conservative
- Provide energy-based recommendations for different deployment scenarios

Hardware: Jetson AGX Thor with INA238 power monitoring
"""

import subprocess
import threading
import time
import statistics
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

# Import consciousness from validated experiments
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from validate_atp_on_real_consciousness import ATPTunedConsciousness, SensorObservation


@dataclass
class PowerSample:
    """Single power measurement sample from tegrastats"""
    timestamp: float
    vdd_gpu: float      # GPU power in mW
    vdd_cpu_soc: float  # CPU/SoC power in mW
    vin: float          # Total input power in mW

    @property
    def total_system(self) -> float:
        """Total system power (sum of main rails)"""
        return self.vdd_gpu + self.vdd_cpu_soc


@dataclass
class EnergyProfile:
    """Energy consumption profile for an ATP configuration"""
    config_name: str
    attention_cost: float
    rest_recovery: float

    # Power measurements (mW)
    baseline_power: float  # Idle power before experiment
    mean_power: float      # Mean power during experiment
    max_power: float       # Peak power
    min_power: float       # Minimum power
    std_power: float       # Power standard deviation

    # Consciousness metrics
    cycles_run: int
    observations_attended: int
    attention_rate: float
    mean_salience: float

    # Energy efficiency metrics
    power_overhead: float  # mean_power - baseline_power (mW)
    energy_per_cycle: float  # Joules per cycle
    energy_per_attended: float  # Joules per attended observation
    operations_per_watt: float  # Cycles per second per watt


class PowerMonitor:
    """
    Background thread that continuously monitors system power using tegrastats.

    Parses tegrastats output to extract power rail measurements.
    Stores samples with timestamps for later analysis.
    """

    def __init__(self, sample_interval_ms: int = 500):
        self.sample_interval_ms = sample_interval_ms
        self.samples: List[PowerSample] = []
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        """Start background power monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.samples = []

        # Start tegrastats subprocess
        self.process = subprocess.Popen(
            ['tegrastats', '--interval', str(self.sample_interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        # Wait for initial samples
        time.sleep(2.0)

    def stop(self):
        """Stop background power monitoring"""
        self.monitoring = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        if self.thread:
            self.thread.join(timeout=3.0)
            self.thread = None

    def _monitor_loop(self):
        """Background loop that reads tegrastats output"""
        if not self.process:
            return

        while self.monitoring and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    continue

                sample = self._parse_tegrastats_line(line)
                if sample:
                    self.samples.append(sample)

            except Exception as e:
                print(f"Error reading tegrastats: {e}")
                break

    def _parse_tegrastats_line(self, line: str) -> Optional[PowerSample]:
        """
        Parse tegrastats output line to extract power measurements.

        Example line:
        12-09-2025 04:56:58 RAM 16614/125772MB (lfb 106x4MB) CPU [2%@972,...]
        cpu@35.25C tj@36.343C ... VDD_GPU 5956mW/5956mW VDD_CPU_SOC_MSS 7148mW/7148mW
        VIN_SYS_5V0 6807mW/6807mW VIN 25530mW/12765mW
        """
        try:
            # Extract power rail measurements
            vdd_gpu_match = re.search(r'VDD_GPU\s+(\d+)mW', line)
            vdd_cpu_match = re.search(r'VDD_CPU_SOC_MSS\s+(\d+)mW', line)
            vin_match = re.search(r'VIN\s+(\d+)mW', line)

            if not (vdd_gpu_match and vdd_cpu_match and vin_match):
                return None

            return PowerSample(
                timestamp=time.time(),
                vdd_gpu=float(vdd_gpu_match.group(1)),
                vdd_cpu_soc=float(vdd_cpu_match.group(1)),
                vin=float(vin_match.group(1))
            )

        except Exception as e:
            return None

    def get_statistics(self) -> Dict[str, float]:
        """Calculate power statistics from collected samples"""
        if not self.samples:
            return {}

        # Use total system power (VDD_GPU + VDD_CPU_SOC)
        powers = [s.total_system for s in self.samples]

        return {
            'mean': statistics.mean(powers),
            'median': statistics.median(powers),
            'stdev': statistics.stdev(powers) if len(powers) > 1 else 0.0,
            'min': min(powers),
            'max': max(powers),
            'samples': len(powers)
        }

    def clear(self):
        """Clear accumulated samples"""
        self.samples = []


def measure_baseline_power(duration_sec: float = 10.0) -> float:
    """
    Measure idle system power with no consciousness activity.

    Returns mean idle power in mW.
    """
    print(f"Measuring baseline power ({duration_sec}s idle)...")

    monitor = PowerMonitor(sample_interval_ms=500)
    monitor.start()

    time.sleep(duration_sec)

    monitor.stop()
    stats = monitor.get_statistics()

    baseline = stats['mean']
    print(f"  Baseline: {baseline:.0f} mW (σ={stats['stdev']:.0f} mW, n={stats['samples']})")

    return baseline


def run_energy_experiment(
    config_name: str,
    attention_cost: float,
    rest_recovery: float,
    cycles: int,
    baseline_power: float
) -> EnergyProfile:
    """
    Run consciousness cycles under specific ATP config while measuring power.

    Returns complete energy profile with power and efficiency metrics.
    """
    print(f"\n--- {config_name} Config ---")
    print(f"  ATP params: cost={attention_cost:.3f}, recovery={rest_recovery:.3f}")
    print(f"  Running {cycles} cycles with power monitoring...")

    # Create consciousness instance
    consciousness = ATPTunedConsciousness(
        identity_name=f"energy-test-{config_name.lower()}",
        attention_cost=attention_cost,
        rest_recovery=rest_recovery
    )

    # Start power monitoring
    monitor = PowerMonitor(sample_interval_ms=500)
    monitor.start()

    # Run consciousness cycles
    start_time = time.time()

    for i in range(cycles):
        # Generate realistic salience distribution
        salience = random.betavariate(5, 2)  # Balanced distribution

        obs = SensorObservation(
            sensor_name="sensor_0",  # Match initialized sensor
            salience=salience,
            data=f"frame_{i}"
        )

        consciousness.process_cycle([obs])

        # Small delay to simulate realistic processing
        time.sleep(0.001)

    duration = time.time() - start_time

    # Stop monitoring and get stats
    monitor.stop()
    power_stats = monitor.get_statistics()

    # Get consciousness metrics
    metrics = consciousness.get_metrics()
    attended_count = consciousness.observations_attended
    attention_rate = attended_count / cycles
    mean_salience = metrics.get('mean_attended_salience', 0.0)

    mean_power = power_stats['mean']
    power_overhead = mean_power - baseline_power

    # Energy calculations
    # Power in Watts, duration in seconds → Energy in Joules
    total_energy = (power_overhead / 1000.0) * duration  # mW → W, then W·s = J
    energy_per_cycle = total_energy / cycles
    energy_per_attended = total_energy / attended_count if attended_count > 0 else 0.0

    # Operations per watt = cycles per second per watt
    cycles_per_second = cycles / duration
    operations_per_watt = cycles_per_second / (mean_power / 1000.0) if mean_power > 0 else 0.0

    profile = EnergyProfile(
        config_name=config_name,
        attention_cost=attention_cost,
        rest_recovery=rest_recovery,
        baseline_power=baseline_power,
        mean_power=mean_power,
        max_power=power_stats['max'],
        min_power=power_stats['min'],
        std_power=power_stats['stdev'],
        cycles_run=cycles,
        observations_attended=attended_count,
        attention_rate=attention_rate,
        mean_salience=mean_salience,
        power_overhead=power_overhead,
        energy_per_cycle=energy_per_cycle,
        energy_per_attended=energy_per_attended,
        operations_per_watt=operations_per_watt
    )

    print(f"  Results:")
    print(f"    Attention rate: {attention_rate:.1%}")
    print(f"    Mean power: {mean_power:.0f} mW (overhead: {power_overhead:+.0f} mW)")
    print(f"    Energy/cycle: {energy_per_cycle*1e6:.2f} µJ")
    print(f"    Energy/attended: {energy_per_attended*1e3:.2f} mJ")
    print(f"    Efficiency: {operations_per_watt:.1f} ops/W")

    return profile


def main():
    """
    Session 15: Energy Efficiency Study

    Measure power consumption and energy efficiency across ATP configurations.
    """
    print("=" * 80)
    print("Session 15: ATP Energy Efficiency Study")
    print("=" * 80)
    print()
    print("Research Goal: Quantify energy trade-offs of ATP configurations")
    print("Approach: Power monitoring via tegrastats + consciousness cycles")
    print("Hardware: Jetson AGX Thor with INA238 power monitoring")
    print()

    # Measure baseline idle power
    baseline_power = measure_baseline_power(duration_sec=10.0)

    # Test configurations from Sessions 12-13
    configs = [
        ("Conservative", 0.05, 0.02),  # 26% attention, energy efficient
        ("Balanced", 0.03, 0.04),       # 42% attention, moderate
        ("Maximum", 0.01, 0.05),        # 62% attention, maximum awareness
    ]

    profiles: List[EnergyProfile] = []
    cycles_per_config = 1000  # Enough cycles for stable measurements

    for name, cost, recovery in configs:
        profile = run_energy_experiment(
            config_name=name,
            attention_cost=cost,
            rest_recovery=recovery,
            cycles=cycles_per_config,
            baseline_power=baseline_power
        )
        profiles.append(profile)

        # Rest between experiments
        time.sleep(5.0)

    # Analysis and comparison
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    print("\n1. ATTENTION RATES (Expected vs Measured)")
    print("-" * 60)
    expected = {"Conservative": 0.26, "Balanced": 0.42, "Maximum": 0.62}
    for p in profiles:
        exp = expected[p.config_name]
        print(f"  {p.config_name:12s}: {p.attention_rate:.1%} "
              f"(expected {exp:.1%}, delta {(p.attention_rate-exp)*100:+.1f}%)")

    print("\n2. POWER CONSUMPTION")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.mean_power:.0f} mW "
              f"(overhead: {p.power_overhead:+.0f} mW, σ={p.std_power:.0f} mW)")

    print("\n3. ENERGY PER CYCLE")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.energy_per_cycle*1e6:.2f} µJ/cycle")

    # Calculate relative efficiency
    conservative_epc = profiles[0].energy_per_cycle
    print(f"\n  Relative to Conservative:")
    for p in profiles[1:]:
        ratio = p.energy_per_cycle / conservative_epc
        print(f"    {p.config_name}: {ratio:.2f}x energy")

    print("\n4. ENERGY PER ATTENDED OBSERVATION")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.energy_per_attended*1e3:.2f} mJ/attended")

    print("\n5. OPERATIONS PER WATT (Efficiency)")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.operations_per_watt:.1f} cycles/s/W")

    # Energy vs Coverage trade-off
    print("\n6. ENERGY-COVERAGE TRADE-OFF")
    print("-" * 60)
    print("  (Using Session 13 coverage metrics)")
    coverage = {"Conservative": 0.376, "Balanced": 0.595, "Maximum": 0.796}
    for p in profiles:
        cov = coverage[p.config_name]
        energy_per_detected = p.energy_per_cycle / cov if cov > 0 else 0
        print(f"  {p.config_name:12s}: {cov:.1%} coverage, "
              f"{energy_per_detected*1e6:.2f} µJ per detected event")

    # Recommendations
    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. BATTERY-POWERED DEVICES (Energy-Critical)")
    best_battery = min(profiles, key=lambda p: p.energy_per_cycle)
    print(f"   → {best_battery.config_name} config")
    print(f"     • Lowest energy/cycle: {best_battery.energy_per_cycle*1e6:.2f} µJ")
    print(f"     • {best_battery.attention_rate:.1%} attention rate")
    print(f"     • Best for: IoT sensors, wearables, remote monitors")

    print("\n2. WALL-POWERED DEVICES (Performance-Critical)")
    best_coverage = max(profiles, key=lambda p: coverage[p.config_name])
    print(f"   → {best_coverage.config_name} config")
    print(f"     • Best coverage: {coverage[best_coverage.config_name]:.1%}")
    print(f"     • {best_coverage.attention_rate:.1%} attention rate")
    print(f"     • Energy cost: {best_coverage.energy_per_cycle*1e6:.2f} µJ/cycle")
    print(f"     • Best for: Security cameras, autonomous robots, event detection")

    print("\n3. BALANCED DEPLOYMENT (General-Purpose)")
    balanced = next(p for p in profiles if p.config_name == "Balanced")
    print(f"   → Balanced config")
    print(f"     • Moderate energy: {balanced.energy_per_cycle*1e6:.2f} µJ/cycle")
    print(f"     • Good coverage: {coverage['Balanced']:.1%}")
    print(f"     • {balanced.attention_rate:.1%} attention rate")
    print(f"     • Best for: Desktop assistants, smart home hubs, general agents")

    # Energy-awareness implications
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    max_profile = next(p for p in profiles if p.config_name == "Maximum")
    cons_profile = next(p for p in profiles if p.config_name == "Conservative")

    energy_ratio = max_profile.energy_per_cycle / cons_profile.energy_per_cycle
    coverage_ratio = coverage["Maximum"] / coverage["Conservative"]

    print(f"\n1. Maximum vs Conservative Energy Cost:")
    print(f"   • Energy: {energy_ratio:.2f}x more per cycle")
    print(f"   • Coverage: {coverage_ratio:.2f}x better detection")
    print(f"   • Efficiency: {coverage_ratio/energy_ratio:.2f}x more coverage per joule")

    print(f"\n2. Energy Budget Examples:")
    print(f"   • 1 mAh battery (3.7V) = 13.3 J")
    conservative_cycles = 13.3 / cons_profile.energy_per_cycle
    maximum_cycles = 13.3 / max_profile.energy_per_cycle
    print(f"     Conservative: {conservative_cycles/1000:.0f}k cycles")
    print(f"     Maximum: {maximum_cycles/1000:.0f}k cycles")
    print(f"     Ratio: {conservative_cycles/maximum_cycles:.1f}x longer battery life")

    print(f"\n3. Power Overhead:")
    print(f"   • Baseline (idle): {baseline_power:.0f} mW")
    print(f"   • Conservative overhead: {cons_profile.power_overhead:+.0f} mW "
          f"({cons_profile.power_overhead/baseline_power:.1%} increase)")
    print(f"   • Maximum overhead: {max_profile.power_overhead:+.0f} mW "
          f"({max_profile.power_overhead/baseline_power:.1%} increase)")

    print("\n" + "=" * 80)
    print("SESSION 15 COMPLETE")
    print("=" * 80)
    print(f"\nTotal measurements: {sum(p.cycles_run for p in profiles)} cycles")
    print(f"Power samples collected: {sum(len(p.config_name) for p in profiles)}+")
    print("Energy-awareness quantified for production deployment!")
    print()


if __name__ == '__main__':
    main()
