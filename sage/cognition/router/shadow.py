#!/usr/bin/env python3
"""
Router shadow-mode hook — Phase 0 Track 5.
==========================================

``RouterShadowHook`` is the integration seam between the existing
consciousness loop and the Phase 0 router data pipeline. It captures
every dispatch decision (as a ``(RouterInput → RouterOutput)`` pair)
without affecting the kernel.

Shadow mode means: the programmatic dispatcher STILL drives actual
dispatch. The hook observes, builds a ``RouterRecord``, applies
SNARC-stratified sampling (PRD §4.7.D), and hands the record to the
dataset writer. Nothing the hook does is allowed to alter kernel state.
Nothing the hook does is allowed to break the kernel.

Design invariants (binding per sprint doc + PRD):

  1. **Failure isolation is airtight**. ``record_decision`` catches
     every exception, logs a warning, and returns ``None``. No caller
     needs a try/except around it — but the caller still wraps its
     integration-layer code in a belt-and-suspenders try/except per
     the sprint doc "pipeline failures must NEVER break the
     consciousness loop" mandate.

  2. **Default OFF**. The hook only runs when
     ``SAGE_ROUTER_SHADOW=1`` is set. No surprise activation.

  3. **Read-only on the kernel**. The hook does NOT mutate WM, SNARC,
     metabolic, plugin registry, or any other kernel state. It only
     reads + writes to the on-disk dataset partition.

  4. **Idempotent across loop ticks**. No accumulating state beyond
     what writer + sampler legitimately need.

  5. **No latency regression** beyond +2ms p99 on the consciousness
     loop (acceptance criterion per sprint doc).

  6. **No torch dependency**. The data pipeline must run on Sprout
     edge with no torch overhead.


Phase 0 known simplifications (documented here, addressed later)
----------------------------------------------------------------

Per the Track 3 PR review gaps:

* **List → single-decision mapping**. The existing dispatcher
  (`_select_attention_targets`) returns a *list* of targets (up to
  ``max_active_plugins``). ``RouterOutput`` is ONE decision per tick.
  The Track 3 programmatic baseline picks the top-salience modality's
  first plugin; the shadow hook mirrors that. This is a deliberate
  Phase 0 simplification; multi-target records land in a future
  schema bump.

* **Habit branch is new in the baseline**. The existing dispatcher
  has no habit short-circuit — cerebellum lookup lives elsewhere in
  the loop, and at Phase 0 ``habit_available`` is always False (no
  cerebellum wired into the feature extractor). The baseline's habit
  branch therefore never fires in shadow records; that's expected.
  Real habit wiring lands in a later phase.

* **``habit_id`` placeholder**. When the baseline's habit branch
  *does* fire (in unit tests, not live shadow), it emits
  ``wm_state_key`` as ``habit_id``. Acceptable for Phase 0 — to be
  properly wired via cerebellum integration in a future phase.

* **``rest``/``crisis`` short-circuit**. Not our problem. The shadow
  records what the existing dispatcher actually does in those
  metabolic states. Only ``dream`` short-circuits to noop in the
  baseline; the shadow captures real behavior for other states.


Usage
-----

Normally constructed once (at ``SAGEConsciousness`` init) and reused
across ticks::

    hook = RouterShadowHook(
        writer=writer,
        sampler=sampler,
        pruner=None,          # Track 7 will schedule pruning
        on_event=None,        # Track 8 will consume events
    )

    router_input = extract_router_input(...)
    programmatic_output = programmatic_decide(router_input, plugin_registry)
    record = hook.record_decision(router_input, programmatic_output)


Spec:
  - phase2/brain-arch/thalamic-router-prd.md
    §2.10 (consciousness-loop integration), §3 (schema), §4.7.D (sampling)
  - phase2/brain-arch/router-sprint-1-phase-0.md
    (Track 5)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from sage.cognition.router.inputs import RouterInput
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.router.record import (
    RouterRecord,
    SOURCE_ENV_VAR,
    VALID_RECORD_SOURCES,
)


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Environment gate
# ──────────────────────────────────────────────────────────────────────

# Per sprint doc: "Toggle via env var SAGE_ROUTER_SHADOW=1 (default
# off until verified per machine)". The env var is consulted once at
# hook construction by ``is_shadow_enabled()``; callers gate their
# lazy instantiation on that helper.
SHADOW_ENV_VAR = "SAGE_ROUTER_SHADOW"


def _build_record_metadata() -> dict:
    """Assemble metadata dict for a new RouterRecord at capture time.

    Currently carries `source` from ``SAGE_SESSION_SOURCE`` env var.
    Unknown or absent value → "idle" as a neutral default. The closed
    vocabulary (VALID_RECORD_SOURCES) is enforced by RouterRecord's
    validator — this function coerces unknown env values to "idle"
    rather than letting record construction fail.
    """
    raw = os.environ.get(SOURCE_ENV_VAR, "").strip()
    source = raw if raw in VALID_RECORD_SOURCES else "idle"
    return {"source": source}


def is_shadow_enabled() -> bool:
    """Return True if the shadow hook should be active this session.

    A shadow hook is active iff the environment variable
    ``SAGE_ROUTER_SHADOW`` is set to ``"1"`` (strict match). Any other
    value — including ``"0"``, ``"true"``, ``""``, or unset — keeps
    the hook disabled.

    We use a strict ``"1"`` match rather than truthy parsing because
    (a) the sprint-doc acceptance text is normative and says "=1",
    and (b) unset vs zero is a meaningful distinction for the
    rollout-canary story — misconfigured envs should fail closed.
    """
    return os.environ.get(SHADOW_ENV_VAR, "") == "1"


# ──────────────────────────────────────────────────────────────────────
# RouterShadowHook
# ──────────────────────────────────────────────────────────────────────

class RouterShadowHook:
    """Failure-isolated capture hook for router training records.

    Parameters
    ----------
    writer:
        ``RouterDatasetWriter`` instance. The hook appends captured
        records via ``writer.append(record_dict)``. The writer owns
        disk I/O + gzip + partition rollover; the hook just funnels.
    sampler:
        ``SnarcStratifiedSampler`` instance. Every decision is
        observed for quintile-rate accounting, but only decisions the
        sampler *keeps* get written. PRD §4.7.D is binding — top
        SNARC quintile is always kept, bottom is 5%.
    pruner:
        Optional ``RouterDatasetPruner`` — held purely as a reference
        so a per-machine deployment (Track 7) can schedule nightly
        pruning without rebuilding. The hook itself NEVER invokes
        the pruner (that would run during a consciousness tick; wrong
        latency budget).
    on_event:
        Optional ``Callable[[str, dict], None]`` for observability.
        Fires for ``"captured"`` / ``"dropped_sampler"`` /
        ``"dropped_writer"`` / ``"error"`` events. Every callback
        exception is caught + logged — the observer never breaks the
        kernel. Track 8 wires this to the dashboard.

    Notes
    -----
    * The hook is cheap to construct but stateful: it owns references
      to writer + sampler, both of which carry rolling state (sampler
      window, writer buffer). Constructing one per tick would reset
      both — don't. Construct once at ``SAGEConsciousness.__init__``
      and reuse.
    * ``record_decision`` is the only method the consciousness loop
      calls. It returns the ``RouterRecord`` (on keep) or ``None``
      (on drop or error). Callers never need the return value — it's
      there for unit tests and future Track 6 outcome tracking.
    """

    def __init__(
        self,
        writer: Any,
        sampler: Any,
        pruner: Optional[Any] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        # Light validation — duck-typed so unit tests can stub writer
        # and sampler without importing the real classes.
        if not hasattr(writer, "append") or not callable(writer.append):
            raise TypeError("writer must have a callable append(record)")
        if not hasattr(sampler, "should_keep") or not callable(sampler.should_keep):
            raise TypeError("sampler must have a callable should_keep(snarc)")

        self.writer = writer
        self.sampler = sampler
        self.pruner = pruner  # deliberately unused at tick time
        self.on_event = on_event

        # Stats — useful for Track 8's dashboard and for operator
        # canary validation during fleet rollout. Never required for
        # correctness; purely observability.
        self.decisions_seen: int = 0
        self.decisions_kept: int = 0
        self.decisions_dropped_sampler: int = 0
        self.decisions_dropped_writer: int = 0
        self.errors: int = 0

    # ──────────────────────────────────────────────────────────────
    # Public API — single entrypoint called per tick
    # ──────────────────────────────────────────────────────────────

    def record_decision(
        self,
        router_input: RouterInput,
        programmatic_output: RouterOutput,
    ) -> Optional[RouterRecord]:
        """Capture one (RouterInput → RouterOutput) decision.

        Returns the constructed ``RouterRecord`` if it was written,
        ``None`` if dropped (by sampler or writer) or on error.

        Never raises. Any exception — even one from a broken
        on_event callback — is caught, counted, logged at WARNING,
        and swallowed. Callers still typically wrap the whole shadow
        path in their own try/except per the kernel's failure-isolation
        mandate (belt and suspenders).
        """
        try:
            self.decisions_seen += 1

            # Build the SNARC dict from the RouterInput itself. This
            # keeps the sampler consistent with whatever values the
            # feature extractor actually captured, rather than using
            # a separately-threaded snapshot that could diverge.
            snarc_dict = _snarc_dict_from_input(router_input)

            # SNARC-stratified keep/drop (PRD §4.7.D).
            try:
                keep = bool(self.sampler.should_keep(snarc_dict))
            except Exception as e:  # noqa: BLE001
                logger.warning("router shadow sampler failed: %s", e)
                self.errors += 1
                self._emit("error", {"stage": "sampler", "reason": repr(e)})
                return None

            if not keep:
                self.decisions_dropped_sampler += 1
                self._emit("dropped_sampler", {
                    "tick": getattr(router_input, "tick", None),
                })
                return None

            # Build the record. Defaults fill record_id (uuid4),
            # schema_version, timestamp, machine (later overridden
            # by writer if empty). Metadata carries source stamp from
            # SAGE_SESSION_SOURCE env var (raising / gameplay / idle /
            # interactive). Absent → "idle" as a neutral default.
            metadata = _build_record_metadata()
            try:
                record = RouterRecord(
                    router_input=router_input,
                    router_output=programmatic_output,
                    metadata=metadata,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("router shadow record construction failed: %s", e)
                self.errors += 1
                self._emit("error", {"stage": "record", "reason": repr(e)})
                return None

            # Hand to the writer. The writer itself is
            # failure-isolated (it returns False on drop, never
            # raises), but we still bracket the call defensively.
            try:
                ok = self.writer.append(record)
            except Exception as e:  # noqa: BLE001
                logger.warning("router shadow writer failed: %s", e)
                self.errors += 1
                self._emit("error", {"stage": "writer", "reason": repr(e)})
                return None

            if not ok:
                self.decisions_dropped_writer += 1
                self._emit("dropped_writer", {
                    "tick": getattr(router_input, "tick", None),
                })
                return None

            self.decisions_kept += 1
            self._emit("captured", {
                "tick": getattr(router_input, "tick", None),
                "action": programmatic_output.action,
                "rationale_code": programmatic_output.rationale_code,
            })
            return record

        except Exception as e:  # noqa: BLE001 — top-level belt-and-suspenders
            # Should be unreachable — every inner block catches. Here
            # purely so a bug in THIS method can't break the kernel.
            logger.warning("router shadow unexpected failure: %s", e)
            self.errors += 1
            return None

    # ──────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a snapshot of hook counters for observability."""
        return {
            "decisions_seen": self.decisions_seen,
            "decisions_kept": self.decisions_kept,
            "decisions_dropped_sampler": self.decisions_dropped_sampler,
            "decisions_dropped_writer": self.decisions_dropped_writer,
            "errors": self.errors,
        }

    # ──────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────

    def _emit(self, kind: str, payload: dict) -> None:
        """Fire the on_event callback, absorbing any exception."""
        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("router shadow on_event(%s) raised: %s", kind, e)
            # Do NOT count as an error; the observer is optional.


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _snarc_dict_from_input(router_input: RouterInput) -> dict:
    """Project the SNARC fields of a RouterInput back to a dict.

    The ``SnarcStratifiedSampler`` expects ``{'arousal': ...,
    'conflict': ..., 'reward': ..., 'surprise': ..., 'novelty': ...}``.
    ``RouterInput`` carries the same values as ``snarc_<dim>`` scalar
    fields. Build the dict directly from the RouterInput so the
    sampler and the written record agree on SNARC values even if the
    caller's SNARC reconstruction varies across ticks.
    """
    return {
        "surprise": float(getattr(router_input, "snarc_surprise", 0.0) or 0.0),
        "novelty": float(getattr(router_input, "snarc_novelty", 0.0) or 0.0),
        "arousal": float(getattr(router_input, "snarc_arousal", 0.0) or 0.0),
        "reward": float(getattr(router_input, "snarc_reward", 0.0) or 0.0),
        "conflict": float(getattr(router_input, "snarc_conflict", 0.0) or 0.0),
    }


__all__ = [
    "RouterShadowHook",
    "is_shadow_enabled",
    "SHADOW_ENV_VAR",
]
