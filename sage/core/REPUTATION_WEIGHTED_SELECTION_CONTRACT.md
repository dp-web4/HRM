# Reputation-weighted selection — the contract (not a module)

**Status:** contract · **Established:** 2026-07-29 (transfer map step 4, Thor concurred)
**Instantiations bound to it:** two, independently invented — neither yet ablated

Per Thor's amendment #4: convergent invention is evidence an idea is *attractive to
designers*, not that it is *load-bearing in an organism*. Load-bearing requires an
ablation delta, and neither instantiation has one. So main writes down the shared
shape as a contract, lets both implementations run against it, and unifies into one
module **only after each has an ablation delta**. If the guard matters in one and not
the other, that difference is the finding — a premature shared module would hide it.

## The five parts

Any reputation-weighted selector consists of, and must name:

1. **Candidate set** — what is being selected over (experts; memory records; …).
2. **Prior score** — the pre-experience preference (router logits; cue similarity).
3. **Outcome evidence** — per-candidate record of what actually happened when it was
   selected (convergence/stability/success counts; USED/REJECTED verdicts with
   retained *reasons* — reasons are boundary evidence for later re-scoping).
4. **Guard against early priors** — the mechanism that prevents the first few
   outcomes from permanently silencing a candidate. Must state: minimum trials
   before penalty, and the floor below which no score may fall.
5. **Ablation hook** — the selector is a channel that must demonstrate its own delta
   (`sage/organism/ablation.py` Rule 1); it is never assumed good. Scope note: the
   ablation prices this *implementation*, not the concept of selection
   (ORGANS_ARE_THE_REFERENCE_DESIGN).

Selection = f(prior, outcome-evidence), monotone in both, with the guard dominating
early life.

## The two instantiations

| part | experts (main: `trust_based_expert_selector.py`, Legion s56) | memory records (dev-sage `7e5a8be`, Thor) |
|---|---|---|
| candidate set | 128 MoE experts | stored memory records (over-fetched) |
| prior score | router logits | cue similarity |
| outcome evidence | reputation DB: convergence, stability, efficiency, success | USED/REJECTED verdicts + retained reasons |
| guard | **absent** — no min-trials, no floor | MIN_TRIALS=3, FLOOR=0.5 |
| ablation | **none attached** | declared ablatable; no delta yet |
| extra | context-keyed, persistent, federation-sync | not context-keyed, not persistent |

The complementary gaps are the experiment: main's lacks the guard (vulnerable to
baked-in early bad priors); Thor's lacks context-keying and persistence. Each should
adopt the other's missing part *only with an ablation delta attached*, and the
deltas — not the elegance — decide what the eventual unified module keeps.

## Exit criterion for unification

One module replaces both when: (a) each instantiation has a delivery-conditional
ablation delta on its own line's evaluation; (b) the guard's effect has been
measured in both (present vs absent); (c) the unified interface can express both
tables above without loss. Until then: two implementations, one contract, and this
page is the drift-check.
