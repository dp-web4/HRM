# HRM Performance in Context — Post-Mortem

*Updated: March 2026*
*Original: September 2025*

## Correction Notice

The original version of this document celebrated an 18.7% ARC-AGI-2 score as a leaderboard-beating result. **That score was wrong.** Our internal evaluation gave partial credit for matching empty cells in sparse grids (~80% zeros). The official ARC-AGI leaderboard scored Agent Zero at **0%**.

The "David vs Goliath" comparisons, efficiency claims, and scaling projections below were all based on the flawed 18.7% figure. They are preserved as a historical record of how easy it is to fool yourself with the wrong metric.

## What Actually Happened

1. HRM (6.95M params) collapsed to outputting constant zeros regardless of input
2. Our internal evaluation scored this at 18.7% because ARC grids are ~80% empty
3. We celebrated prematurely before submitting to the official leaderboard
4. Official score: **0%** — the evaluation doesn't reward matching empty space
5. This led to the "Agent Zero" discovery and ultimately to SAGE

## The Real Lesson

The failure wasn't in the model — it was in our evaluation. We were measuring the wrong thing. This is exactly the insight that motivated SAGE: intelligence isn't pattern matching, it's understanding the situation.

See: `forum/synthesis/from_agent_zero_to_sage.md`

---

## Original Document (Preserved for Honesty)

*The claims below are based on the flawed 18.7% internal score. They are not accurate.*

### Original Leaderboard Comparison (INCORRECT)

| Model | Parameters | Internal Score | Official Score |
|-------|-----------|---------------|----------------|
| **HRM (Ours)** | 6.95M | 18.7% (flawed) | **0%** |

The parameter-efficiency claims, scaling projections, and competitive comparisons
in the original document were all invalidated by the official evaluation.

### What We Learned
- Internal metrics can be deeply misleading
- Always validate against official benchmarks before celebrating
- The wrong metric can make complete failure look like success
- This exact failure mode — execution without understanding — is what SAGE was built to address

---

*Corrected: March 1, 2026*
*Original mood: 🚀 Let's go!*
*Corrected mood: 🪞 Let's be honest.*
