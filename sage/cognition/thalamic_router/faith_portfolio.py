"""Faith-candidate portfolio — parallel win-hypotheses with evolving trust.

dp's refinement (2026-05-23): faith models are CANDIDATES; there can/should be
more than one explored in PARALLEL; their relative trust evolves with accumulated
evidence; plausibility = "could this potentially result in a win?"; each
candidate carries OPEN QUESTIONS that get tracked and resolved.

This generalizes the single-hypothesis commit_faith router mode into a portfolio:

  - Each FaithCandidate is an imagined win-path (a WM template / hypothesis) with
    a plausibility prior P(could-win), an evolving `trust` credence, and a set of
    open questions whose answers move trust.
  - Trust updates on CONFIRM / DISCONFIRM evidence only — NEVER decays on the
    silent unrewarded steps of a feedback desert (the faith rule:
    absence-of-confirmation ≠ disconfirmation). This is the per-candidate update
    law for membot's V3 (Andy/Waving-cat Q3) — `trust` IS that accumulated
    Veracity/Validity; swap the in-memory store for the membot entry when the
    activity-log→T3/V3 glue lands.
  - Commitment is allocated across candidates ∝ trust × plausibility (parallel
    exploration, weighted by relative trust) rather than all-or-nothing.
  - Directed exploration: `next_open_question()` returns the open question whose
    resolution best DISCRIMINATES the leading candidates (value-of-information) —
    so exploration resolves the question that most reallocates commitment, not a
    random probe.

Composes with: CBP's L3 WinImaginer (each candidate's imagined win), the
imaginator-as-membot-entity (each candidate = a WM entry carrying its own V3),
and the commit_faith cognitive-mode (reads the portfolio's best candidate +
top open question to frame the cortex). See
`forum/nomad-faith-horizon-2026-05-23.md` + `nomad-faith-membot-q3-connection-2026-05-23.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OpenQuestion:
    """An unresolved test whose answer moves a candidate's trust.
    `pivotal` ~ how much resolving it would shift trust (the VOI weight)."""
    text: str
    pivotal: float = 0.5                 # [0,1] — expected trust swing on resolution
    status: str = "open"                 # open | confirmed | disconfirmed


@dataclass
class FaithCandidate:
    """One parallel win-hypothesis."""
    id: str
    hypothesis: str                       # "could this result in a win?" statement
    plausibility: float = 0.5             # prior P(could-win | this path)
    trust: float = 0.3                    # evolving credence; membot V3 swap point
    open_questions: List[OpenQuestion] = field(default_factory=list)
    confirms: int = 0
    disconfirms: int = 0
    # v0.2 — MRH-scoped veracity (per metacog v0.2 + CBP review). `strategy` is the
    # reusable pattern key whose veracity accumulates across SIMILAR contexts;
    # `mrh` is this candidate's context (a set of tags). Distinct MRHs let the
    # same axis express e.g. path@L3 vs translation@HAL as separate contexts.
    strategy: Optional[str] = None        # reusable pattern key (defaults to id)
    mrh: frozenset = field(default_factory=frozenset)
    # Step 2 (unbounded-born population): provenance for born/compete/die.
    # born_from: where this candidate came from ("seed" | "wm_rule" | "cortex_hyp"
    # | "transfer" | ...). born_at: the play-cycle index it was spawned (0 = seeded
    # at game start). Let the population reflect what was born when and survived.
    born_from: str = "seed"
    born_at: int = 0
    # Dreamer-informed split (2026-05-26): trust = PREDICTIVE credence ("does this
    # candidate predict accurately", moved by the per-step residual). strategic_value
    # = the STRATEGIC critic ("does committing to this candidate lead to a WIN",
    # moved by advance/abandon outcomes). Dreamer keeps world-model (predictive) and
    # critic (strategic value) as separate organs; we had collapsed both into one
    # scalar with the strategic half (plausibility) FROZEN → the tracking≠winning
    # trap (a candidate predicts perfectly but never wins, yet keeps high score).
    # strategic_value defaults to the plausibility prior and is then UNFROZEN by
    # update_strategic_value(). score = predictive × strategic (both now move).
    strategic_value: Optional[float] = None
    strategic_confirms: int = 0           # times committing here led to advance
    strategic_disconfirms: int = 0        # times committing here led to abandon
    # Per-candidate residual history for SAGE_FAITH_INFO_BOUNDED. Last N
    # residuals seen against this rule's predictions. Used to estimate the
    # rule's observation distribution and weight each confirm by its
    # Shannon information content. A rule whose residuals concentrate at a
    # single value (e.g. all 0 — degenerate noop-predicting rule) gets
    # near-zero info per confirm; a rule with diverse residuals gets full
    # weight. See `forum/nomad-dc22-calibration-diagnosis-2026-06-01.md`.
    _recent_resids: List[float] = field(default_factory=list)
    # CUMULATIVE bucket counts since candidate birth (4 buckets — see
    # FaithPortfolio._bucket_resid). Replaces the windowed estimate for FIB
    # because the recent-window oscillates when consecutive observations
    # cluster in one bucket, transiently driving entropy → 0 even for
    # informative rules. Cumulative counts smooth the long-run distribution.
    # Calibration revision after Thor's v36 A/B (cn04 unintended drop)
    # + CBP's 3-way (97% refute drop, levels rotated not added).
    _bucket_cum: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = self.id
        if self.strategic_value is None:
            self.strategic_value = self.plausibility   # prior; unfrozen by outcomes

    def score(self) -> float:
        """Commitment weight basis: PREDICTIVE × STRATEGIC — trust (does it predict)
        × strategic_value (does committing here win). Backward-compatible: until a
        strategic outcome fires, strategic_value == plausibility, so score reduces to
        the historical trust × plausibility."""
        sv = self.strategic_value if self.strategic_value is not None else self.plausibility
        return self.trust * sv

    @property
    def open_q(self) -> List[OpenQuestion]:
        return [q for q in self.open_questions if q.status == "open"]


# ── MRH-scoped veracity (V := V3.Veracity * MRH) ────────────────────────────
# The local stand-in for membot's per-entry V3 (Andy Q3) indexed by MRH (Andy
# Q2). dp's generalization of "reward": veracity = "has this strategy
# demonstrated success in SIMILAR contexts." `levels_completed` is a per-game MRH
# leaf and doesn't transfer; veracity transfers via context similarity. Swap this
# store for the membot WM-entry read when the activity-log→T3/V3 glue lands.

def _mrh_similarity(query: frozenset, stored: frozenset) -> float:
    """Containment of the stored context in the query — a BROAD stored context
    fully present in the query (e.g. {class:logistics} ⊂ {game:x, lvl:y,
    class:logistics}) contributes fully (→1.0); an unrelated context →0. This is
    what makes broad veracity a cold-start PRIOR for a finer, novel context
    (coarse→fine transfer)."""
    if not stored:
        return 1.0                       # the global context applies everywhere (weak prior)
    inter = len(query & stored)
    return inter / len(stored)


class VeracityStore:
    """Accumulates V3.Veracity per (strategy, MRH-context), aggregates by
    similarity. confirm/disconfirm ONLY — never decays on silence (the faith
    rule). This is the membot V3*MRH swap point."""

    def __init__(self, alpha: float = 0.35, beta: float = 0.5,
                 prior: float = 0.3, conf_k: float = 3.0):
        self.alpha, self.beta, self.prior, self.conf_k = alpha, beta, prior, conf_k
        # strategy -> list of [mrh, trust, n_evidence]
        self._store: Dict[str, List[list]] = {}

    def record(self, strategy: str, mrh: frozenset, kind: str) -> None:
        rows = self._store.setdefault(strategy, [])
        for row in rows:
            if row[0] == mrh:
                break
        else:
            row = [mrh, self.prior, 0]
            rows.append(row)
        if kind == "confirm":
            row[1] = row[1] + self.alpha * (1.0 - row[1]); row[2] += 1
        elif kind == "disconfirm":
            row[1] = row[1] * (1.0 - self.beta); row[2] += 1

    def veracity(self, strategy: str, query_mrh: frozenset) -> float:
        """Similarity-weighted, confidence-weighted veracity of `strategy` in
        the relevant horizon around `query_mrh`. Coarse→fine: broad contexts
        seed the prior; exact-context evidence dominates as it accrues."""
        rows = self._store.get(strategy)
        if not rows:
            return self.prior
        num = den = 0.0
        for mrh, trust, n in rows:
            sim = _mrh_similarity(query_mrh, mrh)
            if sim <= 0:
                continue
            w = sim * (n / (n + self.conf_k))   # similarity × evidence-confidence
            num += w * trust
            den += w
        return num / den if den > 0 else self.prior


# Graded-D calibration (Sprout's v25 dep_resid dump, 2026-05-23). A prediction
# miss counts as DISCONFIRMATION only against a predictor that earned trust;
# a miss against a low-confidence predictor is "never modeled this" → EXPLORE
# (no trust move), not abandon. Only ~3/25 games clear the cutoff at 0.8B, so
# most games' trust moves via cross-context V (the store), not local-D — by design.
D_CONF_CUTOFF = 0.45          # dep_confidence floor to trust a residual as evidence
D_DISAMBIG_CUTOFF = 0.5       # dep_disambiguation floor (a 1-of-200 predictor is noise)
D_THETA_MULT = 3.0           # disconfirm if step_resid > 3× the predictor's in-regime baseline
D_THETA_FALLBACK = 4.0      # flat θ (px) when no per-predictor baseline is available


def tau(mrh: frozenset, budget: Optional[float] = None, base: float = 0.45) -> float:
    """Commitment threshold for the persist decision. τ is ONE READOUT of ATP,
    not ATP's home (CBP review 2026-05-23): budget here modulates the
    persist-vs-abandon bar only; other ATP consumers (exploration budget, memory-
    commit, portfolio cardinality, organ allocation) read ATP elsewhere.

    - scarce budget (low) → raise τ (demand more veracity to commit unrewarded effort)
    - broad/established MRH → lower τ slightly (a strategy proven across many
      contexts earns a lower bar; coarse→fine confidence).
    """
    t = base
    if budget is not None:
        # budget in [0,1]; 0 = empty -> +0.25, 1 = ample -> -0.05
        t += 0.25 * (1.0 - max(0.0, min(1.0, budget))) - 0.05
    breadth = len(mrh)
    if breadth >= 3:                      # finer/more-specific context -> slightly higher bar
        t += 0.03
    return max(0.05, min(0.95, t))


class FaithPortfolio:
    """Parallel faith-candidates with evidence-driven trust reallocation."""

    def __init__(self, alpha: float = 0.35, beta: float = 0.5, commit_tau: float = 0.45,
                 veracity_store: Optional["VeracityStore"] = None):
        self.candidates: Dict[str, FaithCandidate] = {}
        self.alpha = alpha               # confirm step (toward 1.0)
        self.beta = beta                 # disconfirm step (toward 0.0)
        self.commit_tau = commit_tau     # gate: best score must clear this to commit
        # Optional MRH-scoped veracity store (membot V3*MRH stand-in). When set,
        # evidence also accrues per (strategy, MRH) for cross-context transfer.
        self.veracity_store = veracity_store

    # ── portfolio management ────────────────────────────────────────────────
    def add(self, cid: str, hypothesis: str, plausibility: float = 0.5,
            trust: float = 0.3, open_questions: Optional[List[str]] = None,
            pivotals: Optional[List[float]] = None,
            strategy: Optional[str] = None, mrh: Optional[frozenset] = None,
            born_from: str = "seed", born_at: int = 0) -> FaithCandidate:
        oqs = []
        for i, q in enumerate(open_questions or []):
            piv = pivotals[i] if pivotals and i < len(pivotals) else 0.5
            oqs.append(OpenQuestion(text=q, pivotal=piv))
        c = FaithCandidate(id=cid, hypothesis=hypothesis,
                           plausibility=max(0.0, min(1.0, plausibility)),
                           trust=max(0.0, min(1.0, trust)), open_questions=oqs,
                           strategy=strategy, mrh=mrh or frozenset(),
                           born_from=born_from, born_at=born_at)
        # Seed trust from accumulated cross-context veracity if a store knows this
        # strategy in a similar context (the cold-start prior / transfer).
        if self.veracity_store is not None:
            c.trust = self.veracity_store.veracity(c.strategy, c.mrh)
        self.candidates[cid] = c
        return c

    # ── evidence (the ONLY thing that moves trust) ──────────────────────────
    def record(self, cid: str, kind: str, question: Optional[str] = None,
               info_weight: float = 1.0) -> None:
        """kind = 'confirm' | 'disconfirm'. Resolves an open question if named.
        Silence (no reward) must NOT call this — that's the faith rule.

        Under `SAGE_FAITH_TRUST_BAYESIAN=1`, after the per-candidate update we
        also apply a softmax-style normalization across the portfolio so that
        candidate trusts form a relative posterior (per the SAGE-WM math:
        `τ_k ← τ_k · exp(−r_k) / Z` Bayesian model comparison). The
        normalization makes candidates COMPETE — one candidate's improving
        evidence raises its relative trust at the others' expense, which is
        exactly what model-comparison precision is supposed to do. Default OFF
        — existing per-candidate independent updates remain canonical.

        Under `SAGE_FAITH_INFO_BOUNDED=1`, confirms use `info_weight` as the
        trust target — accumulation caps at the rule's per-event information
        content. A rule with degenerate (single-mode) residuals carries near-
        zero info per confirm so trust can't climb on vacuous matches; a rule
        with diverse residuals carries full info per confirm so trust
        accumulates normally. Caller supplies info_weight via
        `update_trust_from_residual` from the candidate's _recent_resids.
        Disconfirms remain full-strength regardless. Default OFF preserves
        existing symmetric-update behavior.
        """
        import os as _os_fpb
        c = self.candidates.get(cid)
        if c is None:
            return
        _info_bounded = _os_fpb.environ.get("SAGE_FAITH_INFO_BOUNDED", "") == "1"
        if kind == "confirm":
            if _info_bounded:
                # Trust pulls toward the info-bounded target (the entropy of the
                # rule's observed outcome distribution) — symmetrically. A rule
                # whose residuals concentrate on one bucket has entropy ≈ 0;
                # ongoing confirms then pull trust DOWN toward 0. Early
                # bootstrap (n < min_samples) uses target=1.0; late-stage
                # observations re-pull trust toward the empirical entropy.
                target = max(0.0, min(1.0, info_weight))
                c.trust = c.trust + self.alpha * (target - c.trust)
            else:
                c.trust = c.trust + self.alpha * (1.0 - c.trust)
            c.confirms += 1
            _resolve(c, question, "confirmed")
        elif kind == "disconfirm":
            c.trust = c.trust * (1.0 - self.beta)
            c.disconfirms += 1
            _resolve(c, question, "disconfirmed")
        # Accrue per-(strategy, MRH) veracity for cross-context transfer (membot
        # V3*MRH stand-in). confirm/disconfirm only — silence never reaches here.
        if self.veracity_store is not None and kind in ("confirm", "disconfirm"):
            self.veracity_store.record(c.strategy, c.mrh, kind)
        # SAGE-WM math: post-hoc Bayesian normalization across portfolio.
        if _os_fpb.environ.get("SAGE_FAITH_TRUST_BAYESIAN", "") == "1":
            self._bayesian_normalize()

    def _bayesian_normalize(self, target_avg: float = 0.5) -> None:
        """Normalize candidate trusts as a softmax over the portfolio so they
        form a relative posterior. Per SAGE-WM math foundation
        (forum/nomad-sage-wm-mathematical-foundation-2026-05-31.md): `τ_k`
        as Bayesian model-comparison precision; the trust-weighted mixture
        is the relative posterior over models.

        Implementation: treat current per-candidate trusts as log-likelihood
        proxies, softmax across the portfolio, scale by K * target_avg so
        that the mean trust ≈ target_avg (preserving the scale that downstream
        thresholds like `commit_tau` were calibrated against). One candidate
        can rise to 1.0 if it dominates; the rest decay relative to it.

        Default target_avg=0.5 chosen so existing `commit_tau=0.45` remains
        a meaningful threshold (a single candidate needs trust >0.45 = above
        the portfolio mean to be commit-eligible).
        """
        import math as _math
        cands = list(self.candidates.values())
        K = len(cands)
        if K == 0:
            return
        # Softmax over log-trusts (treat trust as log-likelihood proxy).
        log_taus = [_math.log(max(c.trust, 1e-6)) for c in cands]
        max_log = max(log_taus)
        exps = [_math.exp(lt - max_log) for lt in log_taus]
        total = sum(exps)
        if total <= 0:
            return
        # Scale so the mean of the new trusts is target_avg.
        # Each candidate's normalized weight is exps[i]/total; multiply by
        # K * target_avg to set mean=target_avg; clamp to [0, 1].
        for c, e in zip(cands, exps):
            c.trust = min(1.0, max(0.0, (e / total) * K * target_avg))

    # ── lifecycle: born / prune (Step 2 — unbounded-born population) ─────────
    def prune(self, trust_floor: float = 0.1, min_evals: int = 3,
              protect: Optional[set] = None) -> List[str]:
        """DEATH half of born/compete/die. Remove candidates whose trust has
        COLLAPSED (< trust_floor) after accumulating at least min_evals trust
        updates (confirms + disconfirms) — i.e. a hypothesis that earned evidence
        and lost, not one that simply hasn't been tested (those abstain via
        explore-noop and are kept). Never removes the last candidate, and never
        removes any cid in `protect` (e.g. the held-out fallback — the population
        must always retain a live explore option). Returns the pruned cids.

        Pruning is what makes the population REFLECT what survived: without it the
        portfolio only grows and best() hides the losers; with it, the surviving
        set is the learned answer to "which hypotheses held up on this game."
        """
        protect = protect or set()
        pruned: List[str] = []
        for cid, c in list(self.candidates.items()):
            if len(self.candidates) - len(pruned) <= 1:
                break  # always keep at least one live candidate
            if cid in protect:
                continue
            evals = c.confirms + c.disconfirms
            if evals >= min_evals and c.trust < trust_floor:
                del self.candidates[cid]
                pruned.append(cid)
        return pruned

    # ── allocation / commitment ─────────────────────────────────────────────
    def weights(self) -> Dict[str, float]:
        """Normalized commitment allocation ∝ trust × plausibility."""
        scored = {cid: c.score() for cid, c in self.candidates.items()}
        tot = sum(scored.values())
        if tot <= 0:
            n = len(scored) or 1
            return {cid: 1.0 / n for cid in scored}
        return {cid: s / tot for cid, s in scored.items()}

    def best(self) -> Optional[FaithCandidate]:
        return max(self.candidates.values(), key=lambda c: c.score(), default=None)

    def best_score(self) -> float:
        b = self.best()
        return b.score() if b else 0.0

    def should_commit(self) -> bool:
        """Gate: is any candidate worth committing to (best score ≥ τ)?"""
        return self.best_score() >= self.commit_tau

    # ── strategic critic update (Dreamer-informed split) ────────────────────
    def update_strategic_value(self, cid: str, outcome: str, lr: float = 0.3) -> str:
        """Move a candidate's STRATEGIC value from the OUTCOME of committing to it —
        the value signal the feedback desert *does* provide:
          'advance' → a level cleared while committed here → EMA toward 1.0 (confirm)
          'abandon' → efficiency-cap / level-reset / game-end without advance while
                      committed here → EMA toward 0.0 (disconfirm)
        The reward-free analog of Dreamer's critic: it learns which candidates actually
        WIN, separately from which merely PREDICT (trust). This is what kills
        tracking≠winning — a high-trust candidate that always leads to abandon sees its
        strategic_value (and thus its score) collapse, so the agent stops committing to
        it. Returns the applied outcome, or 'noop' for unknown cid/outcome.
        """
        c = self.candidates.get(cid)
        if c is None:
            return "noop"
        sv = c.strategic_value if c.strategic_value is not None else c.plausibility
        if outcome == "advance":
            c.strategic_value = max(0.0, min(1.0, sv + lr * (1.0 - sv)))
            c.strategic_confirms += 1
        elif outcome == "abandon":
            c.strategic_value = max(0.0, min(1.0, sv + lr * (0.0 - sv)))
            c.strategic_disconfirms += 1
        else:
            return "noop"
        return outcome

    # ── info-bounded helpers (SAGE_FAITH_INFO_BOUNDED) ──────────────────────
    @staticmethod
    def _bucket_resid(r: float) -> int:
        """Coarse bucket for residual values used by info-weight estimation.
        Buckets: [<0.5px, <2px, <5px, >=5px]. Captures the noop/small/medium/
        large outcome classes that distinguish degenerate from informative
        prediction patterns. Threshold of 0.5px chosen to bucket "effectively
        no change" together (sub-pixel jitter and exact noop both go here)."""
        if r < 0.5:
            return 0
        if r < 2.0:
            return 1
        if r < 5.0:
            return 2
        return 3

    def _info_weight_for_observation(self, c: FaithCandidate,
                                     observed_resid: float,
                                     min_samples: int = 5) -> float:
        """Information weight = Shannon entropy of the candidate's CUMULATIVE
        residual distribution since birth, capped at 1 bit. Used as the trust
        target — trust can only accumulate up to the entropy of the rule's
        observed outcomes over its lifetime.

        **Calibration revision 2026-06-02** after Thor v36 A/B + CBP 3-way
        finding that the 30-event windowed estimator dropped cn04 (informative)
        alongside dc22 (degenerate). The windowed entropy oscillates when
        consecutive observations cluster in one bucket — transiently driving
        entropy → 0 even for an informative rule. Cumulative counts smooth
        the long-run distribution; informative rules' long-run entropy stays
        near the cap, while degenerate rules' long-run entropy stays near 0.

        A rule with degenerate residuals (all concentrated in one bucket, e.g.
        always 0 for a noop-predicting rule on a noop-heavy game) has cumulative
        entropy ≈ 0 → trust can't climb. A rule with diverse residuals (mixed
        0 / 2 / 4px) has cumulative entropy ≈ 1+ bits → trust climbs to full.

        Bootstrap window bumped to 5 (from 3) — gives the cumulative
        distribution more samples before the bound engages. Below min_samples
        info_weight = 1.0 lets candidates bootstrap.
        """
        import math as _math
        cum = c._bucket_cum
        n = sum(cum)
        if n < min_samples:
            return 1.0
        # Shannon entropy of the cumulative bucket distribution
        entropy = 0.0
        for cnt in cum:
            if cnt > 0:
                p = cnt / n
                entropy -= p * _math.log2(p)
        # Cap at 1 bit (matches the target range expected by record())
        return min(entropy, 1.0)

    # ── per-frame trust update = graded-D (the binding-gap fix) ─────────────
    def update_trust_from_residual(self, cid: str, step_resid_px: float,
                                   dep_confidence: float, dep_disambiguation: float,
                                   baseline_resid: Optional[float] = None) -> str:
        """UNFREEZE TRUST. Drive a candidate's confirm/disconfirm from a per-step
        WM prediction residual + the predictor's confidence — Sprout's graded-D.

        Sprout/Legion v26-faith engagement (2026-05-24): the wire fires and the
        plausibility seed is differentiated + model-independent, but trust stayed
        frozen at 0.30 (only the seed was connected), capping best_score at 0.255
        so commit_faith was unreachable. This is the per-frame confirm/disconfirm
        the fleet pinned as the single missing wire.

        Rule (Sprout's calibration):
          - predictor NOT trusted (conf < cutoff or disambig < cutoff): a miss is
            "never modeled this" → EXPLORE, no trust move. (2nd not-falsification case.)
          - predictor trusted: θ = D_THETA_MULT × baseline_resid (or flat fallback).
            step_resid > θ → DISCONFIRM (trusted model missed badly → real D).
            step_resid ≤ θ → CONFIRM (trusted model held → strategy premise holds).

        Returns the action taken: "confirm" | "disconfirm" | "explore-noop".
        NOTE: this is the portfolio-side wire. Live use needs the WM to expose a
        PER-STEP residual+conf at decision time (Sprout's caveat — v25 dep_resid
        is an in-regime aggregate); that accessor is the integration step.
        """
        if cid not in self.candidates:
            return "explore-noop"
        trusted = (dep_confidence >= D_CONF_CUTOFF and dep_disambiguation >= D_DISAMBIG_CUTOFF)
        if not trusted:
            return "explore-noop"            # absence of a trustworthy model ≠ disconfirmation
        theta = (D_THETA_MULT * baseline_resid) if baseline_resid else D_THETA_FALLBACK
        kind = "disconfirm" if step_resid_px > theta else "confirm"
        # Track residual on the candidate for info-weight estimation
        # (SAGE_FAITH_INFO_BOUNDED). CUMULATIVE bucket counts since birth
        # (calibration revision 2026-06-02 — windowed estimator was too noisy,
        # dropped informative rules transiently). Also keep _recent_resids
        # populated for downstream diagnostic tooling.
        c = self.candidates[cid]
        c._recent_resids.append(float(step_resid_px))
        if len(c._recent_resids) > 30:
            c._recent_resids = c._recent_resids[-30:]
        # Info weight for confirms (disconfirms always full-strength). Computed
        # BEFORE the current observation is incorporated into the cumulative
        # distribution — i.e., the prior cumulative distribution weights this
        # confirm. Then increment _bucket_cum AFTER.
        info_w = 1.0
        if kind == "confirm":
            info_w = self._info_weight_for_observation(c, step_resid_px)
        # Increment the cumulative bucket count for this observation
        c._bucket_cum[self._bucket_resid(step_resid_px)] += 1
        self.record(cid, kind, info_weight=info_w)
        return kind

    # ── MRH-scoped veracity reads (v0.2) ────────────────────────────────────
    def veracity_of(self, cid: str, query_mrh: Optional[frozenset] = None) -> float:
        """MRH-scoped veracity of a candidate's strategy. Store-backed when a
        VeracityStore is attached (aggregates over similar contexts); falls back
        to the candidate's own evolving trust otherwise."""
        c = self.candidates.get(cid)
        if c is None:
            return 0.0
        if self.veracity_store is not None:
            return self.veracity_store.veracity(c.strategy, query_mrh if query_mrh is not None else c.mrh)
        return c.trust

    def should_commit_in(self, mrh: frozenset, budget: Optional[float] = None) -> bool:
        """Persist gate at a specific MRH with budget-modulated τ (τ is one
        readout of ATP — see metacog v0.2). Best candidate's MRH-scoped
        veracity × plausibility must clear τ(mrh, budget)."""
        best = None
        best_v = -1.0
        for c in self.candidates.values():
            v = self.veracity_of(c.id, mrh) * c.plausibility
            if v > best_v:
                best_v, best = v, c
        return best is not None and best_v >= tau(mrh, budget, self.commit_tau)

    # ── directed exploration (value-of-information) ─────────────────────────
    def next_open_question(self) -> Optional[Tuple[str, OpenQuestion]]:
        """The open question whose resolution best DISCRIMINATES the leaders.

        VOI heuristic: weight each open question by its `pivotal` AND by how
        contested the portfolio is around its owner — resolving a question on a
        near-tie between the top candidates reallocates the most commitment.
        Returns (candidate_id, OpenQuestion) or None if nothing open.
        """
        ranked = sorted(self.candidates.values(), key=lambda c: -c.score())
        if not ranked:
            return None
        top = ranked[0].score()
        runner = ranked[1].score() if len(ranked) > 1 else 0.0
        contention = 1.0 - (top - runner)          # near-tie -> high contention
        best_q = None
        best_val = -1.0
        for c in ranked:
            for q in c.open_q:
                # leaders' pivotal questions in a contested field score highest
                val = q.pivotal * (0.5 + 0.5 * contention) * (0.5 + 0.5 * c.score())
                if val > best_val:
                    best_val = val
                    best_q = (c.id, q)
        return best_q

    def summary(self) -> List[Dict]:
        w = self.weights()
        return [{"id": c.id, "hypothesis": c.hypothesis,
                 "plausibility": round(c.plausibility, 3), "trust": round(c.trust, 3),
                 "score": round(c.score(), 3), "weight": round(w[c.id], 3),
                 "open_questions": [q.text for q in c.open_q],
                 "confirms": c.confirms, "disconfirms": c.disconfirms}
                for c in sorted(self.candidates.values(), key=lambda c: -c.score())]


def _resolve(c: FaithCandidate, question: Optional[str], status: str) -> None:
    if not question:
        return
    for q in c.open_questions:
        if q.status == "open" and (q.text == question or question in q.text):
            q.status = status
            return
