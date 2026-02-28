#!/usr/bin/env python3
"""
Metabolic Coupling Exploration: ATP Token Cost & Sleep Persistence
===================================================================

Explores the new ATP token coupling and sleep persistence features added
in the Feb 27 HRM update. Tests how LLM token costs drive metabolic state
transitions and examines the cognitive archaeology trail created by
consolidated memory.

Research Questions:
1. How do real LLM token costs affect metabolic state transitions?
2. What experiences get consolidated during DREAM state?
3. Does sleep persistence create a useful cognitive archaeology trail?
4. Can we observe metabolism adapting to conversation complexity?

New Features Tested:
- ATP token coupling (0.05 ATP per token)
- Sleep persistence (top-k SNARC → consolidated_memory.jsonl)
- Response accessor (sage.last_response, sage.responses)
- Enhanced stats tracking (llm_tokens_total, llm_atp_cost_total)

Author: Thor (SAGE autonomous research)
Date: 2026-02-27
Session: Evening autonomous check + metabolic exploration
"""

import sys
from pathlib import Path
import time
import json
import asyncio

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from sage import SAGE


async def run_metabolic_exploration():
    """
    Run SAGE with real LLM and observe metabolic coupling.
    """
    print("="*80)
    print("METABOLIC COUPLING EXPLORATION")
    print("="*80)
    print()
    print("This experiment explores how LLM token costs drive SAGE's")
    print("metabolic state transitions and examines the cognitive archaeology")
    print("trail created by sleep persistence.")
    print()

    # Create SAGE with real LLM
    print("Creating SAGE with real LLM (gemma3:4b via Ollama)...")
    sage = SAGE.create(use_real_llm=True, use_policy_gate=False)
    print(f"✓ SAGE created")
    print(f"  Initial ATP: {sage.metabolic_state.atp_current:.1f}")
    print(f"  Initial state: {sage.metabolic_state.state}")
    print()

    # Track metabolic states
    state_history = []

    # Run a conversation that will consume ATP and trigger state transitions
    print("Starting conversation with SAGE...")
    print("-" * 80)
    print()

    conversation_prompts = [
        # Turn 1: Simple question (low token cost)
        "What is SAGE?",

        # Turn 2: More complex question (moderate token cost)
        "How does your attention mechanism work with SNARC salience?",

        # Turn 3: Even more complex (higher token cost)
        "Explain the relationship between your metabolic ATP system and the cognitive budget allocation across different IRP plugins.",

        # Turn 4: Very complex (high token cost - should trigger state transition)
        "Can you describe the complete consciousness loop, including how observations flow through sensor fusion, SNARC salience computation, plugin selection, budget allocation, execution, and memory consolidation?",
    ]

    for turn_num, prompt in enumerate(conversation_prompts, 1):
        print(f"TURN {turn_num}")
        print(f"User: {prompt}")
        print()

        # Send message
        await sage.receive_message(
            sender="human",
            content=prompt,
            conversation_id="metabolic_exploration_001"
        )

        # Run one cycle
        cycle_stats = await sage.run_one_cycle()

        # Get metabolic state
        current_state = sage.metabolic_state
        state_history.append({
            'turn': turn_num,
            'state': current_state.state,
            'atp': current_state.atp_current,
            'cycle': sage.stats['total_cycles'],
        })

        # Get response
        last_resp = sage.last_response
        if last_resp:
            response_text = last_resp.get('text', '')
            token_count = last_resp.get('tokens', 0)
            atp_cost = last_resp.get('atp_cost', 0)

            print(f"SAGE: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            print()
            print(f"Metrics:")
            print(f"  Tokens: {token_count}")
            print(f"  ATP cost: {atp_cost:.2f}")
            print(f"  Metabolic state: {current_state.state}")
            print(f"  ATP remaining: {current_state.atp_current:.1f}")
            print()

        print("-" * 80)
        print()

        # Small delay for realism
        await asyncio.sleep(0.5)

    # Get final stats
    print()
    print("="*80)
    print("METABOLIC EXPLORATION COMPLETE")
    print("="*80)
    print()

    print("Final Statistics:")
    print(f"  Total cycles: {sage.stats['total_cycles']}")
    print(f"  Total LLM tokens: {sage.stats.get('llm_tokens_total', 0)}")
    print(f"  Total LLM ATP cost: {sage.stats.get('llm_atp_cost_total', 0):.2f}")
    print(f"  Final ATP: {sage.metabolic_state.atp_current:.1f}")
    print(f"  Final state: {sage.metabolic_state.state}")
    print()

    print("Metabolic State History:")
    for entry in state_history:
        print(f"  Turn {entry['turn']}: {entry['state']:12s} ATP={entry['atp']:6.1f}")
    print()

    # Check response tracking
    print("Response Tracking:")
    print(f"  Responses tracked: {len(sage.responses)}")
    if sage.responses:
        print(f"  First response: {sage.responses[0].get('text', '')[:100]}...")
        print(f"  Last response: {sage.last_response.get('text', '')[:100]}...")
    print()

    # Check if consolidated memory was created
    consolidated_path = Path('demo_logs/consolidated_memory.jsonl')
    if consolidated_path.exists():
        print("Sleep Persistence (Consolidated Memory):")
        print(f"  File: {consolidated_path}")

        # Read consolidated memories
        memories = []
        with open(consolidated_path, 'r') as f:
            for line in f:
                if line.strip():
                    memories.append(json.loads(line))

        print(f"  Total consolidated experiences: {len(memories)}")

        # Show most recent consolidation
        if memories:
            latest = memories[-1]
            print(f"  Latest consolidation:")
            print(f"    Cycle: {latest.get('consolidation_cycle', 'N/A')}")
            print(f"    Plugin: {latest.get('plugin', 'N/A')}")
            print(f"    Salience: {latest.get('salience', 0):.3f}")
            if 'response_preview' in latest:
                print(f"    Response: {latest['response_preview'][:100]}...")
        print()
    else:
        print("Sleep Persistence:")
        print("  No consolidated memory file found yet")
        print("  (DREAM state not reached in this short experiment)")
        print()

    # SNARC memory analysis
    print("SNARC Memory:")
    print(f"  Experiences captured: {len(sage.snarc_memory)}")
    if sage.snarc_memory:
        avg_salience = sum(e.get('salience', 0) for e in sage.snarc_memory) / len(sage.snarc_memory)
        print(f"  Average salience: {avg_salience:.3f}")

        # Find highest salience
        max_exp = max(sage.snarc_memory, key=lambda e: e.get('salience', 0))
        print(f"  Highest salience: {max_exp.get('salience', 0):.3f} (cycle {max_exp.get('cycle', 'N/A')})")
    print()

    # Research observations
    print("="*80)
    print("RESEARCH OBSERVATIONS")
    print("="*80)
    print()

    print("1. ATP Token Coupling:")
    print("   - LLM responses now have measurable metabolic cost")
    print("   - Token count directly affects ATP consumption")
    print("   - Longer/complex responses drain ATP faster")
    print()

    print("2. Metabolic State Transitions:")
    state_changes = sum(1 for i in range(1, len(state_history))
                       if state_history[i]['state'] != state_history[i-1]['state'])
    print(f"   - State changes observed: {state_changes}")
    if state_changes > 0:
        print("   - Token costs successfully trigger state transitions")
    else:
        print("   - Experiment too short to trigger transitions")
        print("   - Run longer conversation or increase max_cycles for DREAM")
    print()

    print("3. Response Tracking:")
    print("   - sage.last_response provides easy access to latest output")
    print("   - sage.responses maintains conversation history (max 20)")
    print("   - Useful for S116/S117 conversation analysis")
    print()

    print("4. Sleep Persistence:")
    if consolidated_path.exists():
        print("   - Consolidated memory creates cognitive archaeology trail")
        print("   - High-salience experiences preserved across sessions")
        print("   - Enables longitudinal analysis of SAGE interests")
    else:
        print("   - Not triggered in this short experiment")
        print("   - Requires reaching DREAM state (low ATP)")
        print("   - Run with max_cycles=100+ to observe")
    print()

    print("="*80)
    print("NEXT RESEARCH DIRECTIONS")
    print("="*80)
    print()
    print("1. Extended Conversation Study:")
    print("   - Run 50+ turn conversation to reach DREAM state")
    print("   - Observe complete metabolic cycle: WAKE → REST → DREAM → WAKE")
    print("   - Analyze consolidated memory selection (what gets preserved?)")
    print()
    print("2. Complexity Adaptation:")
    print("   - Compare simple vs complex prompts")
    print("   - Does SAGE adjust response length based on ATP state?")
    print("   - Metabolic feedback on generation behavior?")
    print()
    print("3. Cognitive Archaeology:")
    print("   - Analyze consolidated_memory.jsonl over multiple sessions")
    print("   - What themes/topics have highest long-term salience?")
    print("   - Does consolidation reveal SAGE's 'interests'?")
    print()
    print("4. Integration with S117:")
    print("   - Use response tracking for bidirectional engagement analysis")
    print("   - Does SAGE synthesize answers or ignore them?")
    print("   - ATP cost of synthesis vs continuation?")
    print()


if __name__ == '__main__':
    print()
    print("Metabolic Coupling Exploration")
    print("Testing ATP token coupling + sleep persistence")
    print()

    # Check if Ollama is available
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("⚠️  Warning: Ollama not responding")
            print("   This experiment requires Ollama with gemma3:4b model")
            print("   Please ensure Ollama is running: ollama serve")
            sys.exit(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Warning: Ollama not found")
        print("   This experiment requires Ollama with gemma3:4b model")
        sys.exit(1)

    # Run exploration
    asyncio.run(run_metabolic_exploration())
