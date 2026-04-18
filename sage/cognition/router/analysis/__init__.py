"""Router analysis tooling — post-capture cross-stream joins.

Forward-looking utilities that turn captured router records into
training-ready datasets by joining against external artifacts (game
traces, raising sessions, R6 audit).

The PRD scopes router records to structured kernel state — deliberately
excludes raw environmental inputs (§5). These tools reconstitute the
"did this decision lead to a win?" signal by joining on timestamp.

That signal is what makes Phase 4+ (RPE-grounded online learning) based
on real external outcomes instead of synthetic rewards — and what makes
"can Gemma play the games better?" a directly computable metric.
"""

from sage.cognition.router.analysis.outcome_join import (
    GameSession,
    OutcomeJoiner,
    JoinSummary,
    enrich_records,
)

__all__ = [
    "GameSession",
    "OutcomeJoiner",
    "JoinSummary",
    "enrich_records",
]
