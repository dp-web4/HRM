"""
Extended Deployment Test - SNARC Consciousness with Thresholds
===============================================================

Runs unified SNARC consciousness for extended period (30-60 minutes) to validate:
- Metabolic state transitions over time
- ATP dynamics in sustained operation
- Attention patterns (ATTEND vs IGNORE) across states
- Threshold adaptation to changing conditions
- Resource conservation behavior
- Memory consolidation patterns

This is Option C from thor_worklog.txt recommendations.

Expected: Real-world validation of compression-action-threshold pattern under
sustained operation with natural metabolic state transitions.
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, CompressionMode
import time
import psutil
import random
import json
from datetime import datetime

# Import from thor_unified_snarc_consciousness
exec(open('thor_unified_snarc_consciousness.py').read())


def create_realistic_sensors():
    """
    Create sensors that produce varying salience to trigger state transitions.
    Higher variation than test scenarios to enable natural WAKE/FOCUS/REST transitions.
    """

    def cpu_sensor():
        """CPU monitoring with realistic variation"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        load_1, load_5, load_15 = psutil.getloadavg()

        # High CPU or load → high salience
        urgent = 1 if cpu_percent > 80 else 0
        novelty = min(1.0, cpu_percent / 100.0 + random.random() * 0.2)

        return {
            'cpu_percent': cpu_percent,
            'load_1': load_1,
            'urgent_count': urgent,
            'novelty_score': novelty,
            'value': cpu_percent
        }

    def memory_sensor():
        """Memory monitoring with realistic variation"""
        memory = psutil.virtual_memory()

        urgent = 1 if memory.percent > 85 else 0
        novelty = min(1.0, memory.percent / 100.0 + random.random() * 0.15)

        return {
            'memory_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'urgent_count': urgent,
            'novelty_score': novelty,
            'value': memory.percent
        }

    def process_sensor():
        """Process count monitoring"""
        proc_count = len(psutil.pids())

        # Process variation → moderate novelty
        base_novelty = 0.2
        variation = (proc_count % 10) / 10.0 * 0.3
        novelty = base_novelty + variation + random.random() * 0.2

        return {
            'count': proc_count,
            'urgent_count': 0,
            'novelty_score': novelty,
            'value': proc_count
        }

    def disk_sensor():
        """Disk I/O monitoring"""
        disk = psutil.disk_usage('/')

        urgent = 1 if disk.percent > 90 else 0
        novelty = min(1.0, disk.percent / 100.0 + random.random() * 0.1)

        return {
            'disk_percent': disk.percent,
            'free_gb': disk.free / (1024**3),
            'urgent_count': urgent,
            'novelty_score': novelty,
            'value': disk.percent
        }

    return {
        'cpu_sensor': cpu_sensor,
        'memory_sensor': memory_sensor,
        'process_sensor': process_sensor,
        'disk_sensor': disk_sensor
    }


def create_action_handlers():
    """Create realistic action handlers"""

    def cpu_action(data):
        return ActionResult(
            description=f"CPU: {data.get('cpu_percent', 0):.1f}% (load={data.get('load_1', 0):.2f})",
            reward=0.9,
            trust_validated=True
        )

    def memory_action(data):
        return ActionResult(
            description=f"Memory: {data.get('memory_percent', 0):.1f}% ({data.get('available_gb', 0):.1f}GB avail)",
            reward=0.8,
            trust_validated=True
        )

    def process_action(data):
        return ActionResult(
            description=f"Processes: {data.get('count', 0)}",
            reward=0.7,
            trust_validated=random.random() > 0.15  # 85% success rate
        )

    def disk_action(data):
        return ActionResult(
            description=f"Disk: {data.get('disk_percent', 0):.1f}% ({data.get('free_gb', 0):.0f}GB free)",
            reward=0.75,
            trust_validated=True
        )

    return {
        'cpu_sensor': cpu_action,
        'memory_sensor': memory_action,
        'process_sensor': process_action,
        'disk_sensor': disk_action
    }


def analyze_session_data(consciousness):
    """Analyze extended deployment session data"""

    print("\n" + "="*80)
    print("EXTENDED DEPLOYMENT ANALYSIS")
    print("="*80)

    history = consciousness.execution_history

    if not history:
        print("No execution history to analyze")
        return {}

    # Metabolic state analysis
    states = {}
    for h in history:
        state = h['state']
        states[state] = states.get(state, 0) + 1

    print("\n1. Metabolic State Distribution:")
    total_cycles = len(history)
    for state, count in sorted(states.items()):
        pct = (count / total_cycles) * 100
        print(f"   {state.upper():8s}: {count:4d} cycles ({pct:5.1f}%)")

    # Attention decision analysis
    attended = sum(1 for h in history if h.get('attended', True))
    ignored = total_cycles - attended
    attention_rate = attended / total_cycles if total_cycles > 0 else 0

    print(f"\n2. Attention Decisions:")
    print(f"   Total cycles: {total_cycles}")
    print(f"   ATTENDED: {attended} ({attention_rate*100:.1f}%)")
    print(f"   IGNORED: {ignored} ({(1-attention_rate)*100:.1f}%)")

    # ATP analysis
    atp_values = [h.get('atp', 1.0) for h in history]
    if atp_values:
        avg_atp = sum(atp_values) / len(atp_values)
        min_atp = min(atp_values)
        max_atp = max(atp_values)

        print(f"\n3. ATP Dynamics:")
        print(f"   Average ATP: {avg_atp:.3f}")
        print(f"   Min ATP: {min_atp:.3f}")
        print(f"   Max ATP: {max_atp:.3f}")
        print(f"   Final ATP: {consciousness.atp_remaining:.3f}")

    # Threshold analysis
    threshold_values = [h.get('threshold', 0.5) for h in history]
    if threshold_values:
        avg_threshold = sum(threshold_values) / len(threshold_values)
        min_threshold = min(threshold_values)
        max_threshold = max(threshold_values)

        print(f"\n4. Threshold Dynamics:")
        print(f"   Average threshold: {avg_threshold:.3f}")
        print(f"   Min threshold: {min_threshold:.3f}")
        print(f"   Max threshold: {max_threshold:.3f}")

    # Salience analysis
    salience_values = [h['salience'] for h in history]
    if salience_values:
        avg_salience = sum(salience_values) / len(salience_values)
        min_salience = min(salience_values)
        max_salience = max(salience_values)

        print(f"\n5. Salience Distribution:")
        print(f"   Average salience: {avg_salience:.3f}")
        print(f"   Min salience: {min_salience:.3f}")
        print(f"   Max salience: {max_salience:.3f}")

    # Attention rate by state
    print(f"\n6. Attention Rate by Metabolic State:")
    for state in ['wake', 'focus', 'rest', 'dream']:
        state_history = [h for h in history if h['state'] == state]
        if state_history:
            state_attended = sum(1 for h in state_history if h.get('attended', True))
            state_total = len(state_history)
            state_rate = state_attended / state_total if state_total > 0 else 0
            print(f"   {state.upper():8s}: {state_attended}/{state_total} ({state_rate*100:.1f}%)")

    # State transitions
    transitions = consciousness.state_transitions
    print(f"\n7. Metabolic State Transitions: {len(transitions)}")
    for trans in transitions[-10:]:  # Last 10 transitions
        print(f"   {trans}")

    # Trust evolution
    print(f"\n8. Trust Evolution:")
    for sensor_id in ['cpu_sensor', 'memory_sensor', 'process_sensor', 'disk_sensor']:
        lct_id = f"lct:thor:{sensor_id.replace('_sensor', '')}"
        score = consciousness.trust_oracle.get_full_score(lct_id)
        print(f"   {sensor_id:15s}: T3={score.t3_score():.3f}, V3={score.v3_score():.3f}, " +
              f"Composite={score.composite_score():.3f} ({score.successful_observations}/{score.total_observations})")

    # Memory analysis
    memories = consciousness.memories
    if memories:
        avg_strength = sum(m.strength for m in memories) / len(memories)
        avg_trust = sum(m.trust_score for m in memories) / len(memories)
        avg_salience = sum(m.salience for m in memories) / len(memories)

        print(f"\n9. Memory Quality:")
        print(f"   Total memories: {len(memories)}")
        print(f"   Average strength: {avg_strength:.3f}")
        print(f"   Average trust: {avg_trust:.3f}")
        print(f"   Average salience: {avg_salience:.3f}")

    print("\n" + "="*80)

    # Return analysis data
    return {
        'total_cycles': total_cycles,
        'states': states,
        'attended': attended,
        'ignored': ignored,
        'attention_rate': attention_rate,
        'avg_atp': avg_atp if atp_values else 0,
        'avg_threshold': avg_threshold if threshold_values else 0,
        'avg_salience': avg_salience if salience_values else 0,
        'transitions': len(transitions),
        'final_atp': consciousness.atp_remaining
    }


# ============================================================================
# Main Extended Deployment Test
# ============================================================================

if __name__ == "__main__":

    duration_minutes = 30  # Default 30 minutes

    if len(sys.argv) > 1:
        try:
            duration_minutes = int(sys.argv[1])
        except:
            print(f"Usage: {sys.argv[0]} [duration_minutes]")
            sys.exit(1)

    duration_seconds = duration_minutes * 60

    print("="*80)
    print("EXTENDED DEPLOYMENT TEST - SNARC CONSCIOUSNESS WITH THRESHOLDS")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Duration: {duration_minutes} minutes ({duration_seconds} seconds)")
    print(f"  Platform: Thor (Jetson AGX)")
    print(f"  Pattern: Compression-Action-Threshold (complete)")
    print()
    print("Features Enabled:")
    print("  - SNARC compression (Surprise, Novelty, Arousal, Reward, Conflict)")
    print("  - Trust-weighted attention (Web4 T3/V3)")
    print("  - Metabolic-state-dependent thresholds")
    print("  - ATP energy system (consumption + regeneration)")
    print("  - Binary ATTEND/IGNORE decisions")
    print("  - DREAM consolidation")
    print("  - Cross-session persistence")
    print()
    print("Monitoring:")
    print("  - Metabolic state transitions")
    print("  - ATP dynamics")
    print("  - Attention decision patterns")
    print("  - Threshold modulation")
    print("  - Memory consolidation")
    print()

    # Initialize trust oracle
    oracle = TrustOracle()

    # Register sensors with realistic trust profiles
    oracle.register_entity("lct:thor:cpu", TrustScore(
        lct_id="lct:thor:cpu",
        talent=0.9, training=0.9, temperament=0.95,
        veracity=0.9, validity=0.9, valuation=0.85
    ))

    oracle.register_entity("lct:thor:memory", TrustScore(
        lct_id="lct:thor:memory",
        talent=0.8, training=0.7, temperament=0.85,
        veracity=0.8, validity=0.8, valuation=0.75
    ))

    oracle.register_entity("lct:thor:process", TrustScore(
        lct_id="lct:thor:process",
        talent=0.6, training=0.5, temperament=0.7,
        veracity=0.6, validity=0.6, valuation=0.6
    ))

    oracle.register_entity("lct:thor:disk", TrustScore(
        lct_id="lct:thor:disk",
        talent=0.85, training=0.8, temperament=0.9,
        veracity=0.85, validity=0.85, valuation=0.8
    ))

    # Create sensors and actions
    sensors = create_realistic_sensors()
    actions = create_action_handlers()

    # Create consciousness configuration
    config = ConsciousnessConfig(
        session_id=f"extended_deployment_{int(time.time())}",
        memory_limit=100,
        cycle_delay=2.0,  # 2 second cycles
        trust_salience_weight=0.3,
        trust_memory_weight=0.5,
        initial_atp=1.0,
        atp_regeneration_rate=0.05,
        atp_consumption_per_action=0.1,
        task_criticality=0.5,
        enable_logging=True,
        verbose=False,
        status_report_interval=300.0  # Report every 5 minutes
    )

    # Create consciousness
    consciousness = UnifiedTrustConsciousness(
        sensor_sources=sensors,
        action_handlers=actions,
        trust_oracle=oracle,
        sensor_lct_ids={
            'cpu_sensor': 'lct:thor:cpu',
            'memory_sensor': 'lct:thor:memory',
            'process_sensor': 'lct:thor:process',
            'disk_sensor': 'lct:thor:disk'
        },
        config=config
    )

    print(f"Session: {config.session_id}")
    print(f"Database: {config.db_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Initial Trust Scores:")
    for sensor_id in ['cpu_sensor', 'memory_sensor', 'process_sensor', 'disk_sensor']:
        lct_id = f"lct:thor:{sensor_id.replace('_sensor', '')}"
        score = oracle.get_trust_score(lct_id)
        print(f"  {sensor_id:15s}: {score:.3f}")
    print()
    print("Starting extended deployment...")
    print("(Press Ctrl+C for graceful shutdown)")
    print("="*80)
    print()

    # Run consciousness
    start_time = time.time()
    consciousness.run(duration_seconds=duration_seconds)
    end_time = time.time()

    actual_duration = end_time - start_time

    print(f"\n\nExtended deployment completed!")
    print(f"Actual runtime: {actual_duration:.1f} seconds ({actual_duration/60:.1f} minutes)")
    print()

    # Analyze results
    analysis = analyze_session_data(consciousness)

    # Save analysis to file
    analysis_file = f"extended_deployment_{int(start_time)}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {analysis_file}")
    print()
    print("✅ Extended deployment test complete!")
    print("="*80)
