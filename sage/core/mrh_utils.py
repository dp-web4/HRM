#!/usr/bin/env python3
"""
MRH (Markov Relevancy Horizon) Utilities

Utilities for working with MRH profiles and context boundaries.

Based on Web4 MRH+LCT Context Specification:
- ΔR (spatial): local, regional, global
- ΔT (temporal): ephemeral, session, day, epoch
- ΔC (complexity): simple, agent-scale, society-scale
"""

from typing import Dict, List, Tuple, Optional


# MRH dimension definitions
MRH_SPATIAL = ['local', 'regional', 'global']
MRH_TEMPORAL = ['ephemeral', 'session', 'day', 'epoch']
MRH_COMPLEXITY = ['simple', 'agent-scale', 'society-scale']


def compute_mrh_similarity(mrh1: Dict[str, str], mrh2: Dict[str, str]) -> float:
    """
    Compute similarity between two MRH profiles

    Similarity is simple exact match counting:
    - All 3 dimensions match → 1.0 (perfect fit)
    - 2 dimensions match → 0.67
    - 1 dimension matches → 0.33
    - 0 dimensions match → 0.0 (complete mismatch)

    Args:
        mrh1: First MRH profile
        mrh2: Second MRH profile

    Returns:
        Similarity score in [0, 1]

    Example:
        >>> situation_mrh = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
        >>> pattern_mrh = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
        >>> compute_mrh_similarity(situation_mrh, pattern_mrh)
        1.0  # Perfect match
    """
    matches = 0
    total = 0

    for dim in ['deltaR', 'deltaT', 'deltaC']:
        if dim in mrh1 and dim in mrh2:
            total += 1
            if mrh1[dim] == mrh2[dim]:
                matches += 1

    return matches / total if total > 0 else 0.0


def compute_mrh_distance(mrh1: Dict[str, str], mrh2: Dict[str, str]) -> float:
    """
    Compute distance between two MRH profiles

    Distance considers ordinal nature of dimensions:
    - local → regional = 1 step
    - local → global = 2 steps
    - ephemeral → session = 1 step
    - simple → agent-scale → society-scale = 1/2 steps

    Args:
        mrh1: First MRH profile
        mrh2: Second MRH profile

    Returns:
        Normalized distance in [0, 1]

    Example:
        >>> situation = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
        >>> plugin = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}
        >>> compute_mrh_distance(situation, plugin)
        0.33  # 2 steps out of max 6 steps
    """
    total_distance = 0
    max_distance = 0

    # Spatial distance
    if 'deltaR' in mrh1 and 'deltaR' in mrh2:
        idx1 = MRH_SPATIAL.index(mrh1['deltaR']) if mrh1['deltaR'] in MRH_SPATIAL else 0
        idx2 = MRH_SPATIAL.index(mrh2['deltaR']) if mrh2['deltaR'] in MRH_SPATIAL else 0
        total_distance += abs(idx1 - idx2)
        max_distance += len(MRH_SPATIAL) - 1

    # Temporal distance
    if 'deltaT' in mrh1 and 'deltaT' in mrh2:
        idx1 = MRH_TEMPORAL.index(mrh1['deltaT']) if mrh1['deltaT'] in MRH_TEMPORAL else 0
        idx2 = MRH_TEMPORAL.index(mrh2['deltaT']) if mrh2['deltaT'] in MRH_TEMPORAL else 0
        total_distance += abs(idx1 - idx2)
        max_distance += len(MRH_TEMPORAL) - 1

    # Complexity distance
    if 'deltaC' in mrh1 and 'deltaC' in mrh2:
        idx1 = MRH_COMPLEXITY.index(mrh1['deltaC']) if mrh1['deltaC'] in MRH_COMPLEXITY else 0
        idx2 = MRH_COMPLEXITY.index(mrh2['deltaC']) if mrh2['deltaC'] in MRH_COMPLEXITY else 0
        total_distance += abs(idx1 - idx2)
        max_distance += len(MRH_COMPLEXITY) - 1

    return total_distance / max_distance if max_distance > 0 else 0.0


def infer_situation_mrh(text: str, context: Optional[Dict] = None) -> Dict[str, str]:
    """
    Infer MRH profile from situation/query text

    Uses heuristics to guess appropriate MRH:
    - Quick/fast/simple → ephemeral + simple
    - Question words → session + agent-scale (requires reasoning)
    - Greetings/acks → ephemeral + simple
    - Complex/analyze/explain → session + agent-scale
    - Temporal keywords → appropriate temporal extent

    Args:
        text: Situation or query text
        context: Optional context information

    Returns:
        Inferred MRH profile

    Example:
        >>> infer_situation_mrh("Hello!")
        {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}

        >>> infer_situation_mrh("What did we discuss yesterday?")
        {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'}

        >>> infer_situation_mrh("What patterns emerged this month?")
        {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'}
    """
    text_lower = text.lower()
    # Split into words for proper boundary matching
    words = set(text_lower.replace('?', '').replace('!', '').replace(',', '').replace('.', '').split())

    # Default: local spatial (most common for edge AI)
    deltaR = 'local'

    # Infer temporal extent with explicit keyword detection
    deltaT = 'session'  # Default

    # Ephemeral keywords (immediate/current) - use word set for exact matching
    if any(word in words for word in ['hello', 'hi', 'hey', 'ok', 'yes', 'no', 'thanks', 'bye']) or \
       any(phrase in text_lower for phrase in ['just now', 'right now', 'currently']):
        deltaT = 'ephemeral'

    # Day keywords (yesterday, recent past)
    elif any(word in text_lower for word in ['yesterday', 'today', 'last night', 'this morning', 'earlier today', 'accomplished', 'completed recently']):
        deltaT = 'day'

    # Epoch keywords (patterns over time, long-term trends)
    elif any(word in text_lower for word in ['this month', 'this year', 'over time', 'pattern', 'trend', 'emerged', 'learned over', 'long-term']):
        deltaT = 'epoch'

    # Session keywords (recent conversation, hours)
    elif any(word in text_lower for word in ['remember', 'earlier', 'before', 'recently', 'discussed', 'talked about', 'mentioned']):
        deltaT = 'session'

    # Question words default to session
    elif any(word in text_lower for word in ['what', 'why', 'how', 'explain', 'tell me']):
        deltaT = 'session'

    # Infer complexity extent - use word set for exact matching where needed
    if any(word in words for word in ['hello', 'hi', 'status', 'ok', 'thanks']):
        deltaC = 'simple'  # Simple acknowledgments
    elif any(word in text_lower for word in ['pattern', 'trend', 'emerged', 'learned']):
        deltaC = 'society-scale'  # Pattern analysis across time
    elif any(word in words for word in ['analyze', 'explain', 'why', 'how', 'complex', 'reason']):
        deltaC = 'agent-scale'  # Requires reasoning
    elif len(words) <= 3:
        deltaC = 'simple'  # Short queries usually simple
    else:
        deltaC = 'agent-scale'  # Default to agent-scale for longer queries

    # Adjust spatial extent based on temporal extent
    # Day/epoch queries typically access network resources (epistemic DB, blockchain)
    if deltaT in ['day', 'epoch']:
        deltaR = 'regional'  # Network-accessible memory backends

    # Check for web/external references
    if any(word in text_lower for word in ['web', 'internet', 'search', 'online', 'api']):
        deltaR = 'global'

    return {
        'deltaR': deltaR,
        'deltaT': deltaT,
        'deltaC': deltaC
    }


def format_mrh(mrh: Dict[str, str]) -> str:
    """
    Format MRH profile for display

    Args:
        mrh: MRH profile dict

    Returns:
        Formatted string

    Example:
        >>> mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}
        >>> format_mrh(mrh)
        '(local, session, agent-scale)'
    """
    return f"({mrh.get('deltaR', '?')}, {mrh.get('deltaT', '?')}, {mrh.get('deltaC', '?')})"


def select_plugin_with_mrh(
    situation: str,
    plugins: List[Tuple[str, object]],
    trust_scores: Dict[str, float],
    atp_costs: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    mrh_threshold: float = 0.6
) -> Tuple[str, float]:
    """
    Select best plugin considering MRH fit, trust, and ATP cost

    Strategy: "Correctness first, cost second"
    1. Filter plugins by minimum MRH similarity threshold
    2. Among good matches, balance trust and cost
    3. If no good matches, pick best MRH regardless of cost

    Selection score = trust × mrh_similarity × (1 / atp_cost) × weights

    Args:
        situation: Situation or query text
        plugins: List of (plugin_name, plugin_object) tuples
        trust_scores: Trust score for each plugin
        atp_costs: ATP cost for each plugin
        weights: Optional weights for trust, mrh, atp (default equal)
        mrh_threshold: Minimum MRH similarity to consider (default 0.6 = 2/3 dimensions match)

    Returns:
        (selected_plugin_name, selection_score)

    Example:
        >>> plugins = [
        ...     ('pattern', pattern_engine),
        ...     ('qwen', qwen_plugin)
        ... ]
        >>> trust_scores = {'pattern': 0.9, 'qwen': 0.7}
        >>> atp_costs = {'pattern': 1, 'qwen': 10}
        >>> select_plugin_with_mrh("Hello!", plugins, trust_scores, atp_costs)
        ('pattern', 0.9)  # Perfect MRH match + low cost + high trust
    """
    if weights is None:
        weights = {'trust': 1.0, 'mrh': 1.0, 'atp': 1.0}

    # Infer situation MRH
    situation_mrh = infer_situation_mrh(situation)

    # Score all plugins
    scored_plugins = []

    for plugin_name, plugin_obj in plugins:
        # Get plugin MRH profile
        if hasattr(plugin_obj, 'get_mrh_profile'):
            plugin_mrh = plugin_obj.get_mrh_profile()
        else:
            # Default: assume session-scale agent
            plugin_mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

        # Get trust score
        trust = trust_scores.get(plugin_name, 0.5)

        # Get ATP cost
        atp_cost = atp_costs.get(plugin_name, 1.0)

        # Compute MRH similarity
        mrh_similarity = compute_mrh_similarity(situation_mrh, plugin_mrh)

        # Compute selection score
        score = (
            weights['trust'] * trust *
            weights['mrh'] * mrh_similarity *
            weights['atp'] * (1.0 / atp_cost)
        )

        scored_plugins.append((plugin_name, score, mrh_similarity))

    # Strategy: Correctness first, cost second
    # Filter by MRH threshold
    good_matches = [(name, score, mrh) for name, score, mrh in scored_plugins if mrh >= mrh_threshold]

    if good_matches:
        # Pick best score among good matches
        best = max(good_matches, key=lambda x: x[1])
        return (best[0], best[1])
    else:
        # No good matches - pick best MRH regardless of cost
        # (Better to use expensive backend with right horizon than cheap backend with wrong horizon)
        best = max(scored_plugins, key=lambda x: x[2])  # Max MRH similarity
        return (best[0], best[1])


class InsufficientATPBudgetError(Exception):
    """Raised when quality requirements exceed ATP budget"""
    pass


class NoQualifiedPluginError(Exception):
    """Raised when no plugin meets quality requirements"""
    pass


def select_plugin_with_quality_edge(
    situation: str,
    quality_required: float,
    plugins: List[Tuple[str, object]],
    trust_scores: Dict[str, float],
    atp_costs: Dict[str, float],
    mrh_threshold: float = 0.6,
    atp_budget: Optional[float] = None
) -> Tuple[str, float, str]:
    """
    Edge-optimized plugin selection with quality requirements

    Priority order for edge:
    1. MRH fit (horizon match) - must exceed threshold
    2. Quality requirements (MUST meet minimum) - non-negotiable on edge
    3. ATP budget (MUST fit in budget) - hard constraint
    4. Cost optimization (pick cheapest among qualified)

    Edge philosophy: Fail fast with clear errors rather than silently compromise quality.

    Args:
        situation: Situation or query text
        quality_required: Minimum quality threshold (0-1) - uses trust_score as proxy
        plugins: List of (plugin_name, plugin_object) tuples
        trust_scores: Trust score for each plugin (quality proxy)
        atp_costs: ATP cost for each plugin
        mrh_threshold: Minimum MRH similarity (default 0.6 = 2/3 dimensions)
        atp_budget: Maximum ATP allowed (None = unlimited)

    Returns:
        (selected_plugin_name, atp_cost, selection_reason)

    Raises:
        InsufficientATPBudgetError: When quality requirements exceed ATP budget
        NoQualifiedPluginError: When no plugin meets quality requirements

    Example:
        >>> plugins = [('qwen-0.5b', q1), ('qwen-7b', q2), ('gpt-4', gpt)]
        >>> trust_scores = {'qwen-0.5b': 0.7, 'qwen-7b': 0.9, 'gpt-4': 0.95}
        >>> atp_costs = {'qwen-0.5b': 10, 'qwen-7b': 100, 'gpt-4': 200}
        >>> # Complex query requiring high quality (0.9+)
        >>> select_plugin_with_quality_edge(
        ...     "Analyze consciousness implications",
        ...     quality_required=0.9,
        ...     plugins=plugins,
        ...     trust_scores=trust_scores,
        ...     atp_costs=atp_costs
        ... )
        ('qwen-7b', 100, 'quality_match')  # Picks qwen-7b, not cheap qwen-0.5b
    """
    from . import mrh_utils  # For compute_mrh_similarity

    # Infer situation MRH
    situation_mrh = infer_situation_mrh(situation)

    # Score all plugins
    plugin_candidates = []

    for plugin_name, plugin_obj in plugins:
        # Get plugin MRH profile
        if hasattr(plugin_obj, 'get_mrh_profile'):
            plugin_mrh = plugin_obj.get_mrh_profile()
        else:
            plugin_mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

        # Get metrics
        trust = trust_scores.get(plugin_name, 0.5)
        atp_cost = atp_costs.get(plugin_name, 1.0)

        # Compute MRH similarity
        mrh_similarity = compute_mrh_similarity(situation_mrh, plugin_mrh)

        plugin_candidates.append({
            'name': plugin_name,
            'trust': trust,
            'atp_cost': atp_cost,
            'mrh_similarity': mrh_similarity
        })

    # Phase 1: Filter by MRH threshold
    good_mrh = [p for p in plugin_candidates if p['mrh_similarity'] >= mrh_threshold]

    # If no MRH matches, use all candidates but note the degradation
    if not good_mrh:
        good_mrh = plugin_candidates  # Fall back to all

    # Phase 2: Filter by quality requirement (HARD CONSTRAINT on edge)
    good_quality = [p for p in good_mrh if p['trust'] >= quality_required]

    # Phase 3: Filter by ATP budget if specified
    if atp_budget is not None:
        affordable = [p for p in good_quality if p['atp_cost'] <= atp_budget]
    else:
        affordable = good_quality

    # Selection logic with edge-specific behavior
    if affordable:
        # Pick cheapest among qualified candidates
        best = min(affordable, key=lambda p: p['atp_cost'])
        return (best['name'], best['atp_cost'], 'quality_match')

    elif good_quality:
        # Quality candidates exist but exceed ATP budget
        # On edge: fail fast rather than compromise quality
        min_required_cost = min([p['atp_cost'] for p in good_quality])
        raise InsufficientATPBudgetError(
            f"Quality requirement {quality_required:.2f} needs {min_required_cost} ATP, "
            f"but budget is {atp_budget}. Consider reducing quality requirement or increasing budget."
        )

    else:
        # No plugins meet quality requirement
        # On edge: fail fast with clear error
        best_available = max(plugin_candidates, key=lambda p: p['trust'])
        raise NoQualifiedPluginError(
            f"No plugin meets quality requirement {quality_required:.2f}. "
            f"Best available: {best_available['name']} with trust {best_available['trust']:.2f}. "
            f"Consider reducing quality requirement to {best_available['trust']:.2f} or below."
        )


def infer_quality_requirement(query: str) -> float:
    """
    Infer quality requirement from query complexity

    Edge-optimized heuristic for determining minimum quality threshold
    based on query characteristics.

    Args:
        query: The user query text

    Returns:
        Quality requirement (0-1)

    Quality tiers (edge-specific):
        - Simple (0.5): Greetings, acknowledgments, simple factual
        - Medium (0.7): Explanations, summaries, moderate analysis
        - High (0.85): Deep analysis, reasoning, critical tasks
        - Critical (0.95): Safety-critical, financial, legal
    """
    query_lower = query.lower()

    # Critical quality keywords (0.95)
    critical_keywords = ['safety', 'critical', 'urgent', 'important', 'legal', 'financial', 'medical']
    if any(word in query_lower for word in critical_keywords):
        return 0.95

    # High quality keywords (0.85)
    high_keywords = ['analyze', 'explain why', 'deep', 'comprehensive', 'detailed',
                     'implications', 'philosophy', 'consciousness', 'complex']
    if any(word in query_lower for word in high_keywords):
        return 0.85

    # Medium quality keywords (0.7)
    medium_keywords = ['explain', 'describe', 'what is', 'how does', 'why', 'compare',
                       'difference', 'relationship', 'summary']
    if any(word in query_lower for word in medium_keywords):
        return 0.7

    # Simple queries (0.5)
    simple_keywords = ['hello', 'hi', 'hey', 'yes', 'no', 'ok', 'thanks', 'bye']
    if any(word in query_lower for word in simple_keywords):
        return 0.5

    # Length-based heuristic
    word_count = len(query.split())
    if word_count <= 3:
        return 0.5  # Very short = simple
    elif word_count <= 10:
        return 0.7  # Medium length = moderate quality
    else:
        return 0.85  # Long queries usually need higher quality

    return 0.7  # Default: medium quality


def test_mrh_utils():
    """Test MRH utilities"""
    print("="*80)
    print("Testing MRH Utilities")
    print("="*80)
    print()

    # Test 1: Similarity computation
    print("Test 1: MRH Similarity")
    mrh1 = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
    mrh2 = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
    mrh3 = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

    sim_perfect = compute_mrh_similarity(mrh1, mrh2)
    sim_partial = compute_mrh_similarity(mrh1, mrh3)

    print(f"  Perfect match: {sim_perfect:.2f} (expected 1.0)")
    print(f"  Partial match: {sim_partial:.2f} (expected 0.33)")
    print()

    # Test 2: Distance computation
    print("Test 2: MRH Distance")
    dist_same = compute_mrh_distance(mrh1, mrh2)
    dist_diff = compute_mrh_distance(mrh1, mrh3)

    print(f"  Same MRH distance: {dist_same:.2f} (expected 0.0)")
    print(f"  Different MRH distance: {dist_diff:.2f} (expected ~0.33)")
    print()

    # Test 3: Situation inference
    print("Test 3: Situation MRH Inference")
    situations = [
        "Hello!",
        "Can you explain quantum entanglement?",
        "What's the weather like?",
        "Remember what we discussed earlier?"
    ]

    for sit in situations:
        mrh = infer_situation_mrh(sit)
        print(f"  '{sit}'")
        print(f"    → {format_mrh(mrh)}")
    print()

    # Test 4: Plugin selection
    print("Test 4: MRH-Aware Plugin Selection")

    class MockPlugin:
        def __init__(self, mrh):
            self.mrh = mrh
        def get_mrh_profile(self):
            return self.mrh

    pattern_plugin = MockPlugin({'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'})
    qwen_plugin = MockPlugin({'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'})

    plugins = [
        ('pattern', pattern_plugin),
        ('qwen', qwen_plugin)
    ]

    trust_scores = {'pattern': 0.9, 'qwen': 0.7}
    atp_costs = {'pattern': 1, 'qwen': 10}

    test_situations = [
        "Hello!",  # Should favor pattern (perfect MRH match)
        "Explain relativity",  # Should favor qwen (needs reasoning)
    ]

    for sit in test_situations:
        selected, score = select_plugin_with_mrh(sit, plugins, trust_scores, atp_costs)
        sit_mrh = infer_situation_mrh(sit)
        print(f"  '{sit}' → MRH {format_mrh(sit_mrh)}")
        print(f"    Selected: {selected} (score: {score:.3f})")

    print()
    print("✓ All tests complete")


if __name__ == "__main__":
    test_mrh_utils()
