> **Provenance (sage main adoption, 2026-07-29):** principle text VERBATIM from
> dev-sage `d43625b` (Thor + dp), adopted per `TRANSFER_MAP_DEV_SAGE_2026-07.md`.
> Original header bound this to dev-sage's `organism/scan.py`; the lifted
> implementation is `sage/organism/scan.py` (byte-identical). **Method, not
> capability.** Panel Source bindings for MAIN are listed at the end of this file —
> per this doc's own closing rule, an unbound instrument reads U/S, never zero.

# The instrument scan


---

## The principle

> When I learned to fly, I learned the concept of instrument scan. You can't fly a plane IFR by one
> instrument alone. Altimeter gives you altitude, VSI gives you altitude trend, horizon, nav, airspeed —
> **fixating on just one will put you in the ground.** All instruments matter, and together they enable
> situational awareness. A situation may temporarily prioritize one over the others, and that's legitimate,
> but we need to keep scanning the others to know if the situation is changing.
> — dp, 2026-07-29

**Just like all organs matter.** The same argument as `ORGANS_ARE_THE_REFERENCE_DESIGN.md`, applied to
measurement instead of architecture.

This corrects a specific recurring failure — mine — of grabbing one number, optimising it, and when it
disappoints, reaching for a *narrower version of the same number*. That is fixation, and it is how you fly
a working aircraft into terrain.

**The fix is not fewer instruments. It is a scan.**

---

## The panel

| Instrument | Flight analogue | Reads | Source |
|---|---|---|---|
| **COMPOSITION** | attitude indicator | Is the organism whole, connected, insulated? Six-rung ladder, flow in→out→dropped, insulation invariant, object-space gate | liveness json |
| **CAPABILITY** | altimeter | Where we are *now* — accuracy vs the constant-policy baseline | rows jsonl |
| **LEARNING** | VSI | *Trend* — slope across passes. Climbing or sinking | exercise loop |
| **ACTIVITY** | airspeed | Is it doing work? delivery rates, engagements, levels attempted/cleared. **Below stall = not flying** | rows + ladder |
| **COURSE** | nav / heading | Are we going where the PRD says? transfer radius actually achieved | evidence bundles |
| **COORDINATION** | turn coordinator / ball | Are organs cooperating or fighting? contradiction rate, harmful-retrieval rate | rows + blocks |
| **REASONING** | listening to the engine | What the organism *actually said*. Counts are not the signal at low N | run logs |
| **RESERVES** | fuel | GPU hours, wall-clock, budget remaining | run metadata |

**The attitude indicator is the master.** If COMPOSITION is wrong — organs inert, memory contaminating,
representation mismatched — every other reading is describing a different aircraft than the one you're in.

---

## The scan discipline

1. **Every report opens with the full panel.** Not the one instrument that moved.
2. **Prioritising one instrument is legitimate** when the situation calls for it — but say which, say why,
   and keep the others on the sweep.
3. **State the scope line first** (games / levels / N), which is the airspeed-and-altitude of any claim.
4. **A single instrument may never carry a verdict.** It carries a reading.

---

## Cross-check: how a failed instrument is caught

In IFR you catch a dying attitude indicator because it disagrees with the turn coordinator, altimeter and
airspeed. **A silently-wrong instrument reads plausibly; it is caught by disagreement, not by inspection.**
That is precisely our characteristic failure mode (see the 2026-07-28 silent-resolution census — thirteen
failures, all producing well-formed plausible values, none raising).

Known cross-checks, each learned the expensive way:

| Disagreement | Diagnosis |
|---|---|
| CAPABILITY high + COMPOSITION low | scoring by luck or lookup — **sweep 1: warm 55% p<0.001 while 51% of decisions changed with every memory rejected.** The two readings were incompatible; the disagreement was the finding |
| CAPABILITY flat + ACTIVITY high | busy, not progressing |
| ACTIVITY zero on a channel + any verdict about that channel | **not exercised** — the channel delivered nothing to judge (delivery-conditional influence) |
| LEARNING positive + COMPOSITION drifting | the cold control is not flat → leakage; slope is uninterpretable |
| CAPABILITY moving + REASONING unchanged | the number moved for a reason you have not identified — suspect artifact |
| COURSE on-target + LEARNING negative | on heading and descending. Still a crash |

---

## Worked failures this arc — each was single-instrument fixation

- **N=1 memory win** → banked a capability reading with no composition or activity cross-check. Reversed at N=12.
- **N=12 "cold equals baseline"** → reported as a verdict on the stack; reversed at N=416.
- **N=416 "+11pp, p<0.001"** → capability instrument alone; the entire lift was neighbour lookup.
- **"Bare model as baseline"** → proposed removing the attitude indicator to test the altimeter.
- **Move-match narrowing** → when capability disappointed, I proposed a *narrower capability metric*. Textbook fixation.

---

## Operational

`organism/scan.py` emits the panel from run artifacts and prints cross-check warnings where instruments
disagree. Run it on any evidence bundle before writing a report:

```bash
python3 organism/scan.py --rows organism/_v0_out/<run>.rows.jsonl \
                         --liveness organism/_v0_out/<run>.liveness.json
```

Instruments with no source read **U/S** (unserviceable) rather than zero — an instrument that isn't
reporting is not an instrument reading zero, and conflating those is the same error as calling an
unexercised organ inert.

---

## Main's binding status (2026-07-29 — the honest panel)

| Instrument | main source binding | reads |
|---|---|---|
| COMPOSITION | `sage/embodiment/liveness_binding.py` → `~/.sprout/liveness.json` (rungs 1–4 of vision→raising; rungs 5–6 unbound) | partial |
| ACTIVITY | same binding: salient-event + wake-delivery counts, flow in→out→dropped | partial |
| CAPABILITY | no bound source in main | **U/S** |
| LEARNING | no bound source (raising trust/coherence curves exist but are not wired to the panel) | **U/S** |
| COURSE | no bound source | **U/S** |
| COORDINATION | no bound source | **U/S** |
| REASONING | raising session logs exist; not wired | **U/S** |
| RESERVES | no bound source | **U/S** |

An honest panel that is mostly U/S is the point: it shows exactly which instruments
main has never had, instead of reading plausible zeros. Each future binding turns
one U/S live. The `used`/`affected` rungs (sage-daemon experience records; raising
outcome deltas) are the next wiring items.
