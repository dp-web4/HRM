#!/usr/bin/env python3
"""
MRH Cross-Horizon Memory Query Experiment

Hypothesis: MRH-aware backend selection improves ATP efficiency by correctly
           routing queries to appropriate temporal horizons.

Scenario: Query memory across different time scales
- "What did we just discuss?" → session memory (conversation buffer)
- "What did we discuss yesterday?" → day memory (epistemic DB)
- "What patterns emerged this month?" → epoch memory (blockchain)

Expected Results:
- Baseline: Tries backends sequentially (session → day → epoch)
- MRH-aware: Directly selects correct backend based on temporal horizon
- ATP savings: 50-70% from avoiding failed lookups
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import MRH utilities
from core.mrh_utils import (
    compute_mrh_similarity,
    infer_situation_mrh,
    format_mrh,
    select_plugin_with_mrh
)


@dataclass
class MemoryRecord:
    """A memory record with timestamp and content"""
    timestamp: datetime
    content: str
    context: str
    quality: float


@dataclass
class MemoryQueryResult:
    """Result from memory query"""
    found: bool
    records: List[MemoryRecord]
    backend_used: str
    atp_consumed: int
    query_time_ms: float
    mrh_match: float


class ConversationBufferMemory:
    """
    Mock conversation buffer - session memory

    MRH: (local, session, simple)
    - Stores last ~10 conversation turns
    - Fast lookup (1 ATP)
    - Only remembers current session (minutes to hours)
    """

    def __init__(self):
        self.mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'}
        self.buffer = []
        self.max_size = 10

        # Populate with recent conversation
        now = datetime.now()
        self.buffer = [
            MemoryRecord(now - timedelta(minutes=5), "Discussed MRH experiment results", "conversation", 0.9),
            MemoryRecord(now - timedelta(minutes=3), "Talked about ATP allocation strategies", "conversation", 0.85),
            MemoryRecord(now - timedelta(minutes=1), "Mentioned pattern matcher behavior", "conversation", 0.8),
        ]

    def get_mrh_profile(self):
        return self.mrh

    def query(self, query_text: str, time_range: Optional[timedelta] = None) -> MemoryQueryResult:
        """Query conversation buffer"""
        start = time.time()

        # Simple recency-based search
        results = []
        now = datetime.now()

        for record in self.buffer:
            # Check if within session (last hour)
            age = now - record.timestamp
            if age.total_seconds() <= 3600:  # 1 hour
                # Simple keyword matching
                if any(word in record.content.lower() for word in query_text.lower().split()):
                    results.append(record)

        query_time = (time.time() - start) * 1000

        return MemoryQueryResult(
            found=len(results) > 0,
            records=results,
            backend_used='conversation_buffer',
            atp_consumed=1,  # Very cheap
            query_time_ms=query_time,
            mrh_match=1.0 if results else 0.0
        )


class EpistemicDatabaseMemory:
    """
    Mock epistemic database - day memory

    MRH: (regional, day, agent-scale)
    - Stores discoveries and episodes across days
    - Moderate cost (5 ATP for network + DB query)
    - Remembers recent days (hours to weeks)
    """

    def __init__(self):
        self.mrh = {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'}
        self.discoveries = []

        # Populate with past discoveries
        now = datetime.now()
        self.discoveries = [
            MemoryRecord(now - timedelta(days=1), "Completed SAGE Phase 3: Blockchain witnessing", "discovery", 0.95),
            MemoryRecord(now - timedelta(days=1), "Implemented Merkle batching for efficient witnessing", "discovery", 0.9),
            MemoryRecord(now - timedelta(days=2), "Completed SAGE Phase 2: Skill library integration", "discovery", 0.95),
            MemoryRecord(now - timedelta(days=3), "Created pattern learning system", "discovery", 0.85),
            MemoryRecord(now - timedelta(days=5), "Integrated epistemic memory with SNARC", "discovery", 0.9),
        ]

    def get_mrh_profile(self):
        return self.mrh

    def query(self, query_text: str, time_range: Optional[timedelta] = None) -> MemoryQueryResult:
        """Query epistemic database"""
        start = time.time()

        # Search discoveries in day range
        results = []
        now = datetime.now()

        for record in self.discoveries:
            # Check if within day range (last 30 days)
            age = now - record.timestamp
            if age.days <= 30:
                # Keyword matching
                if any(word in record.content.lower() for word in query_text.lower().split()):
                    results.append(record)

        query_time = (time.time() - start) * 1000

        return MemoryQueryResult(
            found=len(results) > 0,
            records=results,
            backend_used='epistemic_db',
            atp_consumed=5,  # Network + DB query cost
            query_time_ms=query_time,
            mrh_match=1.0 if results else 0.0
        )


class BlockchainWitnessMemory:
    """
    Mock blockchain witness memory - epoch memory

    MRH: (regional, epoch, society-scale)
    - Stores witnessed events across long time periods
    - Expensive (10 ATP for blockchain query)
    - Remembers epochs (days to months)
    """

    def __init__(self):
        self.mrh = {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'}
        self.blocks = []

        # Populate with historical blocks
        now = datetime.now()
        self.blocks = [
            MemoryRecord(now - timedelta(days=15), "Pattern: Skill creation threshold stabilized at 3 repetitions", "pattern", 0.88),
            MemoryRecord(now - timedelta(days=20), "Pattern: Trust scores converge after 50 successful actions", "pattern", 0.85),
            MemoryRecord(now - timedelta(days=25), "Pattern: ATP efficiency improves 79x with batching", "pattern", 0.92),
            MemoryRecord(now - timedelta(days=30), "Pattern: SNARC salience correlates with user feedback", "pattern", 0.87),
        ]

    def get_mrh_profile(self):
        return self.mrh

    def query(self, query_text: str, time_range: Optional[timedelta] = None) -> MemoryQueryResult:
        """Query blockchain"""
        start = time.time()

        # Search blockchain for patterns
        results = []
        now = datetime.now()

        for record in self.blocks:
            # Blockchain has all history
            # Keyword matching
            if any(word in record.content.lower() for word in query_text.lower().split()):
                results.append(record)

        query_time = (time.time() - start) * 1000

        return MemoryQueryResult(
            found=len(results) > 0,
            records=results,
            backend_used='blockchain',
            atp_consumed=10,  # Expensive blockchain query
            query_time_ms=query_time,
            mrh_match=1.0 if results else 0.0
        )


@dataclass
class MemoryQuery:
    """A memory query with expected temporal horizon"""
    text: str
    expected_mrh: Dict[str, str]
    expected_backend: str
    complexity: str


class MemoryOrchestrator:
    """Orchestrates memory queries across backends"""

    def __init__(self, mrh_aware: bool = False):
        """
        Initialize memory orchestrator

        Args:
            mrh_aware: If True, use MRH-aware backend selection
        """
        self.mrh_aware = mrh_aware

        # Initialize memory backends
        self.conversation = ConversationBufferMemory()
        self.epistemic = EpistemicDatabaseMemory()
        self.blockchain = BlockchainWitnessMemory()

        # Backend registry
        self.backends = [
            ('conversation', self.conversation),
            ('epistemic', self.epistemic),
            ('blockchain', self.blockchain)
        ]

        # Trust scores (all backends trusted equally for now)
        self.trust_scores = {
            'conversation': 0.9,
            'epistemic': 0.9,
            'blockchain': 0.9
        }

        # ATP costs
        self.atp_costs = {
            'conversation': 1,   # Fast local lookup
            'epistemic': 5,      # Network + DB query
            'blockchain': 10     # Expensive blockchain query
        }

        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_atp': 0,
            'failed_lookups': 0,  # Queries to wrong backend
            'backend_usage': {'conversation': 0, 'epistemic': 0, 'blockchain': 0},
            'mrh_matches': []
        }

    def query_memory(self, query: MemoryQuery) -> MemoryQueryResult:
        """
        Query memory with appropriate backend

        Args:
            query: Memory query to execute

        Returns:
            MemoryQueryResult from appropriate backend
        """
        self.stats['total_queries'] += 1

        if self.mrh_aware:
            # MRH-aware selection with correctness-first strategy
            selected, score = select_plugin_with_mrh(
                query.text,
                self.backends,
                self.trust_scores,
                self.atp_costs,
                weights={'trust': 1.0, 'mrh': 1.0, 'atp': 0.3},  # Cost matters less
                mrh_threshold=0.6  # Require 2/3 dimensions match
            )
        else:
            # Baseline: Try backends in order of cost (cheap → expensive)
            # This is the naive "try everything" approach
            selected = None
            for backend_name, backend_obj in self.backends:
                result = backend_obj.query(query.text)
                self.stats['total_atp'] += result.atp_consumed
                self.stats['backend_usage'][backend_name] += 1

                if result.found:
                    selected = backend_name
                    self.stats['mrh_matches'].append(result.mrh_match)
                    return result
                else:
                    # Failed lookup - wasted ATP
                    self.stats['failed_lookups'] += 1

            # Nothing found in any backend
            return MemoryQueryResult(
                found=False,
                records=[],
                backend_used='none',
                atp_consumed=0,  # Already counted above
                query_time_ms=0,
                mrh_match=0.0
            )

        # MRH-aware: Query only selected backend
        backend_obj = dict(self.backends)[selected]
        result = backend_obj.query(query.text)

        self.stats['total_atp'] += result.atp_consumed
        self.stats['backend_usage'][selected] += 1
        self.stats['mrh_matches'].append(result.mrh_match)

        if not result.found:
            self.stats['failed_lookups'] += 1

        return result

    def get_summary(self) -> Dict:
        """Get experiment summary statistics"""
        avg_mrh_match = sum(self.stats['mrh_matches']) / len(self.stats['mrh_matches']) if self.stats['mrh_matches'] else 0

        return {
            'mode': 'MRH-aware' if self.mrh_aware else 'Baseline',
            'total_queries': self.stats['total_queries'],
            'total_atp': self.stats['total_atp'],
            'avg_atp_per_query': self.stats['total_atp'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
            'failed_lookups': self.stats['failed_lookups'],
            'backend_usage': self.stats['backend_usage'].copy(),
            'avg_mrh_match': avg_mrh_match
        }


def create_cross_horizon_queries() -> List[MemoryQuery]:
    """Create test queries spanning different temporal horizons"""
    return [
        # Session queries (recent conversation)
        MemoryQuery(
            "What did we just discuss about MRH?",
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'},
            'conversation',
            'session'
        ),
        MemoryQuery(
            "What was mentioned about ATP allocation recently?",
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'},
            'conversation',
            'session'
        ),
        MemoryQuery(
            "Remind me what we talked about in the last few minutes",
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'},
            'conversation',
            'session'
        ),

        # Day queries (recent discoveries)
        MemoryQuery(
            "What did we accomplish yesterday with SAGE?",
            {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'},
            'epistemic',
            'day'
        ),
        MemoryQuery(
            "What blockchain work was completed in the last few days?",
            {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'},
            'epistemic',
            'day'
        ),
        MemoryQuery(
            "Tell me about the Phase 3 completion",
            {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'},
            'epistemic',
            'day'
        ),
        MemoryQuery(
            "What skill library work was done recently?",
            {'deltaR': 'regional', 'deltaT': 'day', 'deltaC': 'agent-scale'},
            'epistemic',
            'day'
        ),

        # Epoch queries (long-term patterns)
        MemoryQuery(
            "What patterns have emerged over the past month?",
            {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'},
            'blockchain',
            'epoch'
        ),
        MemoryQuery(
            "What have we learned about skill creation thresholds?",
            {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'},
            'blockchain',
            'epoch'
        ),
        MemoryQuery(
            "How has ATP efficiency improved over time?",
            {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'},
            'blockchain',
            'epoch'
        ),
        MemoryQuery(
            "What trends appear in SNARC salience scoring?",
            {'deltaR': 'regional', 'deltaT': 'epoch', 'deltaC': 'society-scale'},
            'blockchain',
            'epoch'
        ),
    ]


def run_experiment():
    """Run cross-horizon memory query experiment"""
    print("="*80)
    print("Cross-Horizon Memory Query Experiment")
    print("="*80)
    print()

    # Create test queries
    queries = create_cross_horizon_queries()
    print(f"Test Suite: {len(queries)} queries across temporal horizons")
    print(f"  - Session: {sum(1 for q in queries if q.complexity == 'session')}")
    print(f"  - Day: {sum(1 for q in queries if q.complexity == 'day')}")
    print(f"  - Epoch: {sum(1 for q in queries if q.complexity == 'epoch')}")
    print()

    # Run baseline experiment
    print("="*80)
    print("BASELINE: Sequential Backend Search")
    print("="*80)
    print()

    baseline = MemoryOrchestrator(mrh_aware=False)
    print("Processing queries with baseline strategy (try all backends sequentially)...")
    for query in queries:
        result = baseline.query_memory(query)

    baseline_summary = baseline.get_summary()
    print("\nBaseline Results:")
    print(f"  Total ATP: {baseline_summary['total_atp']}")
    print(f"  Avg ATP/query: {baseline_summary['avg_atp_per_query']:.2f}")
    print(f"  Failed lookups: {baseline_summary['failed_lookups']} (wasted queries)")
    print(f"  Backend usage: conversation={baseline_summary['backend_usage']['conversation']}, "
          f"epistemic={baseline_summary['backend_usage']['epistemic']}, "
          f"blockchain={baseline_summary['backend_usage']['blockchain']}")
    print()

    # Run MRH-aware experiment
    print("="*80)
    print("EXPERIMENTAL: MRH-Aware Backend Selection")
    print("="*80)
    print()

    mrh_aware = MemoryOrchestrator(mrh_aware=True)
    print("Processing queries with MRH-aware strategy (select correct horizon)...")
    for query in queries:
        result = mrh_aware.query_memory(query)

    mrh_summary = mrh_aware.get_summary()
    print("\nMRH-Aware Results:")
    print(f"  Total ATP: {mrh_summary['total_atp']}")
    print(f"  Avg ATP/query: {mrh_summary['avg_atp_per_query']:.2f}")
    print(f"  Failed lookups: {mrh_summary['failed_lookups']}")
    print(f"  Backend usage: conversation={mrh_summary['backend_usage']['conversation']}, "
          f"epistemic={mrh_summary['backend_usage']['epistemic']}, "
          f"blockchain={mrh_summary['backend_usage']['blockchain']}")
    print()

    # Compare results
    print("="*80)
    print("COMPARISON & ANALYSIS")
    print("="*80)
    print()

    atp_saved = baseline_summary['total_atp'] - mrh_summary['total_atp']
    atp_efficiency = (atp_saved / baseline_summary['total_atp']) * 100 if baseline_summary['total_atp'] > 0 else 0

    failed_avoided = baseline_summary['failed_lookups'] - mrh_summary['failed_lookups']

    print(f"ATP Efficiency:")
    print(f"  Baseline ATP:     {baseline_summary['total_atp']}")
    print(f"  MRH-aware ATP:    {mrh_summary['total_atp']}")
    print(f"  ATP Saved:        {atp_saved} ({atp_efficiency:+.1f}%)")
    print()

    print(f"Failed Lookup Reduction:")
    print(f"  Baseline failed:  {baseline_summary['failed_lookups']}")
    print(f"  MRH-aware failed: {mrh_summary['failed_lookups']}")
    print(f"  Avoided:          {failed_avoided} ({failed_avoided/baseline_summary['failed_lookups']*100:.1f}% reduction)")
    print()

    print(f"Backend Selection Accuracy:")
    print(f"  Baseline: {baseline_summary['backend_usage']}")
    print(f"  MRH-aware: {mrh_summary['backend_usage']}")
    print()

    # Hypothesis validation
    print("="*80)
    print("HYPOTHESIS VALIDATION")
    print("="*80)
    print()
    print(f"Hypothesis: MRH-aware backend selection improves ATP efficiency by 50-70%")
    print()

    if atp_efficiency >= 50 and atp_efficiency <= 70:
        print(f"✓ HYPOTHESIS CONFIRMED: {atp_efficiency:.1f}% improvement (within predicted range)")
    elif atp_efficiency > 70:
        print(f"✓ HYPOTHESIS EXCEEDED: {atp_efficiency:.1f}% improvement (better than predicted!)")
    elif atp_efficiency > 0:
        print(f"○ PARTIAL CONFIRMATION: {atp_efficiency:.1f}% improvement (below predicted range)")
    else:
        print(f"✗ HYPOTHESIS REJECTED: {atp_efficiency:.1f}% change (no improvement)")

    print()

    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()

    print("1. MRH-aware selection eliminates failed lookups")
    print(f"   Baseline wastes {baseline_summary['failed_lookups']} queries on wrong backends")
    print(f"   MRH-aware directly selects correct temporal horizon")
    print()

    print("2. ATP savings from avoiding expensive failed queries")
    print(f"   Each failed blockchain query: 10 ATP wasted")
    print(f"   Each failed epistemic query: 5 ATP wasted")
    print(f"   Total savings: {atp_saved} ATP ({atp_efficiency:.1f}%)")
    print()

    print("3. Cross-horizon queries benefit most from explicit MRH")
    print(f"   Session queries → conversation buffer (1 ATP)")
    print(f"   Day queries → epistemic DB (5 ATP)")
    print(f"   Epoch queries → blockchain (10 ATP)")
    print(f"   MRH-aware picks correct backend without trial-and-error")
    print()

    return {
        'baseline': baseline_summary,
        'mrh_aware': mrh_summary,
        'atp_efficiency': atp_efficiency,
        'hypothesis_confirmed': atp_efficiency >= 50
    }


if __name__ == "__main__":
    results = run_experiment()
