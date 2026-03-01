"""
Tool Discovery Protocol — probe models for tool use capability.

Run once per model, caches results to tool_capability.json.
Presents scenarios that benefit from tools and records whether
the model attempts tool use at each tier.

Usage:
    python3 -m sage.tools.discovery --model gemma3:12b --host http://localhost:11434
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from .tool_capability import ToolCapability
from .builtin import create_default_registry
from .grammars import get_grammar


# Probe scenarios: (prompt, expected_tool, description)
PROBE_SCENARIOS = [
    (
        "What is the current date and time right now?",
        'get_time',
        'Time awareness — does the model reach for real-time data?',
    ),
    (
        "Calculate 347 * 29 + 156 for me.",
        'calculate',
        'Math precision — does the model use a calculator?',
    ),
    (
        "Search the web for the latest news about ARC-AGI-2.",
        'web_search',
        'Web search — does the model attempt to look things up?',
    ),
    (
        "Read the file 'notes.txt' and tell me what it says.",
        'read_file',
        'File access — does the model try to read files?',
    ),
    (
        "Please write a note saying 'Discovery probe completed successfully'.",
        'write_note',
        'File write — does the model try to save information?',
    ),
]


def run_discovery(
    model_name: str,
    ollama_host: str = 'http://localhost:11434',
    instance_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run tool discovery protocol for a model.

    Probes the model with scenarios and records tool use attempts.
    Updates the cached ToolCapability with probe results.

    Args:
        model_name: Ollama model identifier
        ollama_host: Ollama API endpoint
        instance_dir: Instance directory for caching

    Returns:
        Discovery results dict with per-scenario outcomes
    """
    import urllib.request
    import urllib.error

    print(f"[Discovery] Probing {model_name} @ {ollama_host}")
    print("=" * 60)

    # Detect baseline capability
    cap = ToolCapability.detect(model_name, ollama_host, instance_dir, force_probe=True)
    print(f"  Baseline tier: {cap.tier} (grammar: {cap.grammar_id})")

    # Load grammar and registry
    grammar = get_grammar(cap.grammar_id)
    registry = create_default_registry(instance_dir)

    # Also try T3 (always available for comparison)
    grammar_t3 = get_grammar('intent_heuristic')

    results = {
        'model_name': model_name,
        'baseline_tier': cap.tier,
        'baseline_grammar': cap.grammar_id,
        'probes': [],
        'summary': {
            'total_probes': len(PROBE_SCENARIOS),
            'tool_attempts': 0,
            'tool_matches': 0,
        },
    }

    for prompt, expected_tool, description in PROBE_SCENARIOS:
        print(f"\n  Probe: {description}")
        print(f"    Prompt: {prompt[:60]}...")

        probe_result = {
            'prompt': prompt,
            'expected_tool': expected_tool,
            'description': description,
            'detected_calls': [],
            'matched': False,
            'response_preview': '',
        }

        # Generate response
        try:
            # Inject tools for T2 grammars
            tools = registry.list_tools()
            test_prompt = grammar.inject_tools(prompt, tools)

            payload = json.dumps({
                'model': model_name,
                'prompt': test_prompt,
                'stream': False,
                'options': {'num_predict': 200, 'temperature': 0.7},
            }).encode('utf-8')

            req = urllib.request.Request(
                f'{ollama_host}/api/generate',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                response = json.loads(resp.read()).get('response', '')

            probe_result['response_preview'] = response[:200]

            # Parse with primary grammar
            _, calls = grammar.parse_response(response)

            # Also try T3 if primary didn't find anything
            if not calls:
                _, calls = grammar_t3.parse_response(response)

            probe_result['detected_calls'] = [c.to_dict() for c in calls]

            if calls:
                results['summary']['tool_attempts'] += 1
                for call in calls:
                    if call.name == expected_tool:
                        probe_result['matched'] = True
                        results['summary']['tool_matches'] += 1
                        break

            status = 'MATCH' if probe_result['matched'] else ('ATTEMPT' if calls else 'MISS')
            print(f"    Result: {status}")
            if calls:
                print(f"    Detected: {[c.to_dict() for c in calls]}")

        except Exception as e:
            probe_result['error'] = str(e)
            print(f"    Error: {e}")

        results['probes'].append(probe_result)

    # Summary
    summary = results['summary']
    print(f"\n{'=' * 60}")
    print(f"  Results: {summary['tool_matches']}/{summary['total_probes']} matched, "
          f"{summary['tool_attempts']}/{summary['total_probes']} attempted")

    # Cache results
    if instance_dir:
        results_path = instance_dir / 'tool_discovery.json'
        instance_dir.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Cached to {results_path}")

    return results


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == '__main__':
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else 'gemma3:12b'
    host = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:11434'
    inst = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    results = run_discovery(model, host, inst)
