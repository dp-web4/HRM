#!/usr/bin/env python3
"""
Plugin tier enum for the thalamic router.
=========================================

Per PRD §1.2, every plugin invocation is classified by an ATP cost tier.
The router emits the tier alongside the plugin name so downstream gating
(e.g. step 6 Budget) can refuse high-tier invocations when ATP is low.

Spec: phase2/brain-arch/thalamic-router-prd.md §1.2
"""

from enum import Enum
from typing import List


class PluginTier(str, Enum):
    """ATP cost tier for a plugin invocation.

    Values match the string constants used in `RouterOutput.plugin_tier`
    (PRD §3.2) exactly: 'reflex' | 'routine' | 'specialized' |
    'frontal_lobe' | 'federate'.

    Inheriting from ``str`` makes the enum JSON-serializable by default
    (``json.dumps(PluginTier.REFLEX)`` → ``"reflex"``) and equal to its
    string value so existing dict-comparison code keeps working.

    Tier reference (from PRD §1.2):

    +--------------+-------------------------+--------------+-------------+
    | Tier         | Examples                | Typical ATP  | Latency     |
    +==============+=========================+==============+=============+
    | REFLEX       | habit cache, SNARC      | 0.1 - 1      | < 5 ms      |
    | ROUTINE      | perception filters      | 1 - 10       | 5 - 100 ms  |
    | SPECIALIZED  | game solvers, vision    | 10 - 50      | 50 - 500 ms |
    | FRONTAL_LOBE | LLM reasoning           | 50 - 500     | 0.5 - 10 s  |
    | FEDERATE     | peer SAGE invocation    | 100 - 1000   | 1 - 30 s    |
    +--------------+-------------------------+--------------+-------------+
    """

    REFLEX = "reflex"
    ROUTINE = "routine"
    SPECIALIZED = "specialized"
    FRONTAL_LOBE = "frontal_lobe"
    FEDERATE = "federate"

    @classmethod
    def values(cls) -> List[str]:
        """Return all tier string values in declaration order."""
        return [t.value for t in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check whether ``value`` matches any tier."""
        return value in cls._value2member_map_
