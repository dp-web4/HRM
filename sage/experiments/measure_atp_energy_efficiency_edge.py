#!/usr/bin/env python3
"""
Session 15 Edge Adaptation: ATP Energy Efficiency Study for Jetson Orin Nano

Adapted from Thor's measure_atp_energy_efficiency.py for Sprout's hardware.

Key differences:
- Orin Nano power rails: VDD_IN, VDD_CPU_GPU_CV, VDD_SOC
- Thor power rails: VDD_GPU, VDD_CPU_SOC_MSS, VIN

Hardware: Jetson Orin Nano 8GB (Sprout)
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
    """Single power measurement sample from tegrastats (Orin Nano)"""
    timestamp: float
    vdd_in: float         # Total input power in mW
    vdd_cpu_gpu_cv: float # CPU/GPU/CV power in mW
    vdd_soc: float        # SoC power in mW

    @property
    def total_system(self) -> float:
        """Total system power - use VDD_IN as primary metric"""
        return self.vdd_in


@dataclass
class EnergyProfile:
    """Energy consumption profile for an ATP configuration"""
    config_name: str
    attention_cost: float
    rest_recovery: float

    # Power measurements (mW)
    baseline_power: float
    mean_power: float
    max_power: float
    min_power: float
    std_power: float

    # Consciousness metrics
    cycles_run: int
    observations_attended: int
    attention_rate: float
    mean_salience: float

    # Energy efficiency metrics
    power_overhead: float
    energy_per_cycle: float
    energy_per_attended: float
    operations_per_watt: float


class PowerMonitor:
    """
    Background thread that monitors system power using tegrastats.
    Adapted for Jetson Orin Nano power rails.
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
        Parse tegrastats output for Orin Nano power rails.

        Example line:
        12-09-2025 06:06:03 RAM 4832/7620MB ... VDD_IN 5830mW/5830mW
        VDD_CPU_GPU_CV 1156mW/1156mW VDD_SOC 1594mW/1594mW
        """
        try:
            # Extract Orin Nano power rail measurements
            vdd_in_match = re.search(r'VDD_IN\s+(\d+)mW', line)
            vdd_cpu_gpu_match = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
            vdd_soc_match = re.search(r'VDD_SOC\s+(\d+)mW', line)

            if not (vdd_in_match and vdd_cpu_gpu_match and vdd_soc_match):
                return None

            return PowerSample(
                timestamp=time.time(),
                vdd_in=float(vdd_in_match.group(1)),
                vdd_cpu_gpu_cv=float(vdd_cpu_gpu_match.group(1)),
                vdd_soc=float(vdd_soc_match.group(1))
            )

        except Exception:
            return None

    def get_statistics(self) -> Dict[str, float]:
        """Calculate power statistics from collected samples"""
        if not self.samples:
            return {}

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
    """Measure idle system power with no consciousness activity."""
    print(f"Measuring baseline power ({duration_sec}s idle)...")

    monitor = PowerMonitor(sample_interval_ms=500)
    monitor.start()

    time.sleep(duration_sec)

    monitor.stop()
    stats = monitor.get_statistics()

    if not stats:
        print("  ⚠️ No power samples collected - using estimate")
        return 5800.0  # Typical Orin Nano idle power

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
    """Run consciousness cycles under specific ATP config while measuring power."""
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
        salience = random.betavariate(5, 2)

        obs = SensorObservation(
            sensor_name="sensor_0",
            salience=salience,
            data=f"frame_{i}"
        )

        consciousness.process_cycle([obs])
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

    mean_power = power_stats.get('mean', baseline_power)
    power_overhead = mean_power - baseline_power

    # Energy calculations
    total_energy = (power_overhead / 1000.0) * duration
    energy_per_cycle = total_energy / cycles
    energy_per_attended = total_energy / attended_count if attended_count > 0 else 0.0

    cycles_per_second = cycles / duration
    operations_per_watt = cycles_per_second / (mean_power / 1000.0) if mean_power > 0 else 0.0

    profile = EnergyProfile(
        config_name=config_name,
        attention_cost=attention_cost,
        rest_recovery=rest_recovery,
        baseline_power=baseline_power,
        mean_power=mean_power,
        max_power=power_stats.get('max', mean_power),
        min_power=power_stats.get('min', mean_power),
        std_power=power_stats.get('stdev', 0.0),
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
    """Session 15 Edge: Energy Efficiency Study on Jetson Orin Nano"""
    print("=" * 80)
    print("Session 15 Edge: ATP Energy Efficiency Study")
    print("=" * 80)
    print()
    print("Research Goal: Quantify energy trade-offs of ATP configurations")
    print("Hardware: Jetson Orin Nano 8GB (Sprout)")
    print("Power Rails: VDD_IN, VDD_CPU_GPU_CV, VDD_SOC")
    print()

    # Measure baseline idle power
    baseline_power = measure_baseline_power(duration_sec=10.0)

    # Test configurations
    configs = [
        ("Conservative", 0.05, 0.02),
        ("Balanced", 0.03, 0.04),
        ("Maximum", 0.01, 0.05),
    ]

    profiles: List[EnergyProfile] = []
    cycles_per_config = 1000

    for name, cost, recovery in configs:
        profile = run_energy_experiment(
            config_name=name,
            attention_cost=cost,
            rest_recovery=recovery,
            cycles=cycles_per_config,
            baseline_power=baseline_power
        )
        profiles.append(profile)
        time.sleep(5.0)

    # Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS (Orin Nano 8GB)")
    print("=" * 80)

    print("\n1. ATTENTION RATES")
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

    conservative_epc = profiles[0].energy_per_cycle if profiles[0].energy_per_cycle > 0 else 1e-9
    print(f"\n  Relative to Conservative:")
    for p in profiles[1:]:
        if p.energy_per_cycle > 0:
            ratio = p.energy_per_cycle / conservative_epc
            print(f"    {p.config_name}: {ratio:.2f}x energy")

    print("\n4. ENERGY PER ATTENDED OBSERVATION")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.energy_per_attended*1e3:.2f} mJ/attended")

    print("\n5. OPERATIONS PER WATT")
    print("-" * 60)
    for p in profiles:
        print(f"  {p.config_name:12s}: {p.operations_per_watt:.1f} cycles/s/W")

    # Coverage trade-off
    print("\n6. ENERGY-COVERAGE TRADE-OFF")
    print("-" * 60)
    coverage = {"Conservative": 0.376, "Balanced": 0.595, "Maximum": 0.796}
    for p in profiles:
        cov = coverage[p.config_name]
        energy_per_detected = p.energy_per_cycle / cov if cov > 0 else 0
        print(f"  {p.config_name:12s}: {cov:.1%} coverage, "
              f"{energy_per_detected*1e6:.2f} µJ per detected event")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS (Orin Nano 8GB)")
    print("=" * 80)

    max_profile = next(p for p in profiles if p.config_name == "Maximum")
    cons_profile = next(p for p in profiles if p.config_name == "Conservative")

    if cons_profile.energy_per_cycle > 0 and max_profile.energy_per_cycle > 0:
        energy_ratio = max_profile.energy_per_cycle / cons_profile.energy_per_cycle
        coverage_ratio = coverage["Maximum"] / coverage["Conservative"]

        print(f"\n1. Maximum vs Conservative:")
        print(f"   • Energy: {energy_ratio:.2f}x more per cycle")
        print(f"   • Coverage: {coverage_ratio:.2f}x better detection")
        print(f"   • Efficiency: {coverage_ratio/energy_ratio:.2f}x more coverage per joule")

    print(f"\n2. Power Budget:")
    print(f"   • Baseline (idle): {baseline_power:.0f} mW")
    print(f"   • Conservative overhead: {cons_profile.power_overhead:+.0f} mW")
    print(f"   • Maximum overhead: {max_profile.power_overhead:+.0f} mW")

    print("\n" + "=" * 80)
    print("SESSION 15 EDGE VALIDATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
