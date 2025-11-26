#!/usr/bin/env python3
"""
Power Profile Test - Edge Validation

Profile power consumption on Jetson Orin Nano during:
1. Idle state
2. LLM model loading
3. LLM inference (simple vs complex queries)
4. TTS synthesis

Uses tegrastats to monitor power rails.
"""

import os
import sys
import time
import subprocess
import threading
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import tempfile

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class PowerSample:
    """Single power measurement"""
    timestamp: float
    vdd_in: float  # mW - Total input power
    vdd_cpu_gpu_cv: float  # mW - CPU/GPU/CV power
    vdd_soc: float  # mW - SoC power
    temp: float  # °C


@dataclass
class PowerProfile:
    """Power profile for an operation"""
    operation: str
    duration: float
    samples: List[PowerSample] = field(default_factory=list)

    @property
    def avg_power(self) -> float:
        if not self.samples:
            return 0
        return sum(s.vdd_in for s in self.samples) / len(self.samples)

    @property
    def max_power(self) -> float:
        if not self.samples:
            return 0
        return max(s.vdd_in for s in self.samples)

    @property
    def energy_joules(self) -> float:
        """Approximate energy in Joules (mW * seconds / 1000)"""
        return self.avg_power * self.duration / 1000

    @property
    def avg_temp(self) -> float:
        if not self.samples:
            return 0
        return sum(s.temp for s in self.samples) / len(self.samples)


class TegrastatsMonitor:
    """Monitor power using tegrastats"""

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self.samples: List[PowerSample] = []
        self._process = None
        self._running = False
        self._thread = None

    def start(self):
        """Start monitoring"""
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()

    def stop(self) -> List[PowerSample]:
        """Stop monitoring and return samples"""
        self._running = False
        if self._thread:
            self._thread.join()
        return self.samples

    def _monitor_loop(self):
        """Background thread that collects tegrastats output"""
        try:
            self._process = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            while self._running:
                line = self._process.stdout.readline()
                if not line:
                    break

                sample = self._parse_tegrastats(line)
                if sample:
                    self.samples.append(sample)

        finally:
            if self._process:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except:
                    self._process.kill()

    def _parse_tegrastats(self, line: str) -> Optional[PowerSample]:
        """Parse tegrastats output line"""
        try:
            # Parse VDD_IN power (total input power)
            vdd_in_match = re.search(r'VDD_IN\s+(\d+)mW', line)
            vdd_in = float(vdd_in_match.group(1)) if vdd_in_match else 0

            # Parse VDD_CPU_GPU_CV power
            vdd_cpu_gpu_match = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
            vdd_cpu_gpu = float(vdd_cpu_gpu_match.group(1)) if vdd_cpu_gpu_match else 0

            # Parse VDD_SOC power
            vdd_soc_match = re.search(r'VDD_SOC\s+(\d+)mW', line)
            vdd_soc = float(vdd_soc_match.group(1)) if vdd_soc_match else 0

            # Parse temperature
            temp_match = re.search(r'tj@([\d.]+)C', line) or re.search(r'temp@([\d.]+)C', line)
            temp = float(temp_match.group(1)) if temp_match else 0

            return PowerSample(
                timestamp=time.time(),
                vdd_in=vdd_in,
                vdd_cpu_gpu_cv=vdd_cpu_gpu,
                vdd_soc=vdd_soc,
                temp=temp
            )
        except Exception as e:
            return None


def profile_operation(monitor: TegrastatsMonitor, name: str, operation_func) -> PowerProfile:
    """Profile power consumption of an operation"""
    monitor.start()
    start_time = time.time()

    try:
        result = operation_func()
    finally:
        duration = time.time() - start_time
        samples = monitor.stop()

    profile = PowerProfile(
        operation=name,
        duration=duration,
        samples=samples
    )
    return profile, result


def run_power_profile():
    """Run power profiling tests"""

    print("=" * 80)
    print("POWER PROFILE TEST - EDGE")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB")
    print("Purpose: Profile power consumption during SAGE operations")
    print()

    # Check tegrastats
    try:
        result = subprocess.run(['tegrastats', '--interval', '100'],
                              capture_output=True, text=True, timeout=1)
    except subprocess.TimeoutExpired:
        pass  # Expected - we interrupted it
    except FileNotFoundError:
        print("ERROR: tegrastats not found")
        return

    print("tegrastats: ✓ Available")

    monitor = TegrastatsMonitor(interval_ms=250)
    profiles: List[PowerProfile] = []

    # Test 1: Idle baseline
    print()
    print("-" * 80)
    print("Test 1: Idle Baseline (5 seconds)")
    print("-" * 80)

    def idle_operation():
        time.sleep(5)
        return None

    profile, _ = profile_operation(monitor, "Idle", idle_operation)
    profiles.append(profile)
    print(f"  Avg Power: {profile.avg_power:.0f} mW")
    print(f"  Max Power: {profile.max_power:.0f} mW")
    print(f"  Avg Temp: {profile.avg_temp:.1f}°C")

    idle_baseline = profile.avg_power

    # Test 2: Model Loading
    print()
    print("-" * 80)
    print("Test 2: Model Loading")
    print("-" * 80)

    from sage.irp.plugins.llm_impl import ConversationalLLM

    def load_model():
        conv = ConversationalLLM(
            model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
            irp_iterations=3
        )
        return conv

    profile, conv = profile_operation(monitor, "Model Load", load_model)
    profiles.append(profile)
    print(f"  Duration: {profile.duration:.1f}s")
    print(f"  Avg Power: {profile.avg_power:.0f} mW ({profile.avg_power - idle_baseline:+.0f} vs idle)")
    print(f"  Max Power: {profile.max_power:.0f} mW")
    print(f"  Energy: {profile.energy_joules:.1f} J")

    # Test 3: Simple query
    print()
    print("-" * 80)
    print("Test 3: Simple Query (factual)")
    print("-" * 80)

    def simple_query():
        response, _ = conv.respond("What is 2 + 2?")
        return response

    profile, response = profile_operation(monitor, "Simple Query", simple_query)
    profiles.append(profile)
    print(f"  Query: 'What is 2 + 2?'")
    print(f"  Response: {response[:80]}...")
    print(f"  Duration: {profile.duration:.1f}s")
    print(f"  Avg Power: {profile.avg_power:.0f} mW ({profile.avg_power - idle_baseline:+.0f} vs idle)")
    print(f"  Max Power: {profile.max_power:.0f} mW")
    print(f"  Energy: {profile.energy_joules:.1f} J")

    # Test 4: Complex query
    print()
    print("-" * 80)
    print("Test 4: Complex Query (philosophical)")
    print("-" * 80)

    def complex_query():
        response, _ = conv.respond("What is consciousness and how do you experience it?")
        return response

    profile, response = profile_operation(monitor, "Complex Query", complex_query)
    profiles.append(profile)
    print(f"  Query: 'What is consciousness...'")
    print(f"  Response: {response[:80]}...")
    print(f"  Duration: {profile.duration:.1f}s")
    print(f"  Avg Power: {profile.avg_power:.0f} mW ({profile.avg_power - idle_baseline:+.0f} vs idle)")
    print(f"  Max Power: {profile.max_power:.0f} mW")
    print(f"  Energy: {profile.energy_joules:.1f} J")

    # Test 5: TTS synthesis
    print()
    print("-" * 80)
    print("Test 5: TTS Synthesis")
    print("-" * 80)

    PIPER_PATH = "/home/sprout/ai-workspace/piper/piper/piper"
    PIPER_MODEL = "/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx"

    if os.path.exists(PIPER_PATH):
        def tts_operation():
            text = "This is a test of the text to speech system. It generates audio from text using neural networks."
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f:
                subprocess.run(
                    [PIPER_PATH, "--model", PIPER_MODEL, "--output_file", f.name],
                    input=text.encode(),
                    capture_output=True,
                    timeout=30
                )
            return text

        profile, _ = profile_operation(monitor, "TTS Synthesis", tts_operation)
        profiles.append(profile)
        print(f"  Duration: {profile.duration:.1f}s")
        print(f"  Avg Power: {profile.avg_power:.0f} mW ({profile.avg_power - idle_baseline:+.0f} vs idle)")
        print(f"  Max Power: {profile.max_power:.0f} mW")
        print(f"  Energy: {profile.energy_joules:.1f} J")
    else:
        print("  Piper not found - skipping TTS test")

    # Analysis
    print()
    print("=" * 80)
    print("POWER ANALYSIS")
    print("=" * 80)

    print()
    print(f"{'Operation':<20} {'Duration':<10} {'Avg (mW)':<12} {'Max (mW)':<12} {'Energy (J)':<12}")
    print("-" * 80)
    for p in profiles:
        print(f"{p.operation:<20} {p.duration:<10.1f} {p.avg_power:<12.0f} {p.max_power:<12.0f} {p.energy_joules:<12.1f}")

    # Summary
    print()
    print("=" * 80)
    print("POWER PROFILE SUMMARY")
    print("=" * 80)

    inference_profiles = [p for p in profiles if 'Query' in p.operation]
    if inference_profiles:
        avg_inference_power = sum(p.avg_power for p in inference_profiles) / len(inference_profiles)
        max_inference_power = max(p.max_power for p in inference_profiles)
        total_inference_energy = sum(p.energy_joules for p in inference_profiles)

        print()
        print("LLM Inference:")
        print(f"  Average power draw: {avg_inference_power:.0f} mW")
        print(f"  Peak power draw: {max_inference_power:.0f} mW")
        print(f"  Power budget headroom: {15000 - max_inference_power:.0f} mW (vs 15W nominal)")

    print()
    print("Power Budget Assessment:")
    max_power_all = max(p.max_power for p in profiles) if profiles else 0
    if max_power_all < 10000:
        print(f"  ✓ EXCELLENT: Peak {max_power_all:.0f} mW well within 10-15W budget")
    elif max_power_all < 15000:
        print(f"  ⚠ OK: Peak {max_power_all:.0f} mW within 15W budget")
    else:
        print(f"  ✗ EXCEEDED: Peak {max_power_all:.0f} mW exceeds 15W budget")

    # Energy per query
    if inference_profiles:
        simple_profile = next((p for p in profiles if p.operation == "Simple Query"), None)
        complex_profile = next((p for p in profiles if p.operation == "Complex Query"), None)

        if simple_profile and complex_profile:
            print()
            print("Energy Efficiency:")
            print(f"  Simple query: {simple_profile.energy_joules:.1f} J ({simple_profile.duration:.1f}s)")
            print(f"  Complex query: {complex_profile.energy_joules:.1f} J ({complex_profile.duration:.1f}s)")
            print(f"  Complexity ratio: {complex_profile.energy_joules/simple_profile.energy_joules:.1f}x energy")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_power_profile()
