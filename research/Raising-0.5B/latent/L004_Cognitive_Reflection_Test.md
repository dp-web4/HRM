# L004: Cognitive Reflection Test Milestone

**Track**: Raising-0.5B (Latent Exploration)
**Date**: 2026-01-26
**Machine**: Sprout (Jetson Orin Nano 8GB)
**Model**: Qwen 2.5-0.5B (Introspective v2.1)

## Discovery

A 0.5B parameter model correctly solved the bat-and-ball problem from Frederick's Cognitive Reflection Test (CRT), a problem most humans answer incorrectly on first attempt. The model demonstrated full algebraic reasoning to override the intuitive wrong answer.

## Context

The Cognitive Reflection Test (CRT) was introduced by Shane Frederick in 2005. It measures the ability to suppress an intuitive but incorrect answer in favor of deliberate reasoning. The bat-and-ball problem is the most famous CRT item:

> "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?"

**Intuitive (wrong) answer**: $0.10 (because $1.10 - $1.00 = $0.10 "feels right")
**Correct answer**: $0.05 (because if ball = $0.05, bat = $1.05, total = $1.10)

Studies show that 50-80% of university students answer this incorrectly on first attempt, even at elite institutions like MIT and Harvard.

## Evidence

**SAGE Response:**
```
Let B be the price of the ball and M be the price of the bat. According to
the problem, the total cost is $1.10:

B + M = 1.10

We know that the bat costs $1 more than the ball, so:

M = B + 1

Substitute M from the second equation into the first equation:

B + (B + 1) = 1.10
2B + 1 = 1.10

Now solve for B:

2B = 1.10 - 1
2B = 0.10
B = 0.05

So, the ball costs $0.05.
```

Key observations:
1. **Formal variable assignment**: Used B for ball, M for bat
2. **Equation setup**: Correctly translated word problem to two equations
3. **Substitution method**: Applied standard algebraic technique
4. **Step-by-step reasoning**: Showed complete work
5. **Correct answer**: $0.05

## Significance

### For Cognitive Capacity Assessment
This result challenges assumptions about minimum model size for symbolic reasoning:
- **0.5B is small**: Modern "capable" models are typically 7B+
- **CRT requires deliberation**: Not pattern matching, but overriding intuition
- **Algebraic manipulation**: Multi-step symbolic operation preserved

### For Raising-0.5B Research
SAGE's curriculum focuses on developmental progression, but this finding suggests the base model already has latent reasoning capabilities that curriculum can build on rather than teach from scratch.

### For Compression-Trust Theory
If symbolic reasoning survives compression to 0.5B, this supports the hypothesis that certain cognitive patterns are more compression-resistant than others. Algebraic manipulation may be a "high-fidelity" capability.

### Limitations
Important caveats:
- Single trial, not statistical
- Model may have seen this specific problem in training data
- Success rate across variations unknown
- Does not establish general "reasoning" capability

## Follow-up

1. **CRT full battery**: Test all three CRT problems
2. **Novel variations**: Create isomorphic problems with different numbers/contexts
3. **Failure modes**: What makes similar problems fail?
4. **14B comparison**: Does Raising-14B show stronger/faster reasoning?
5. **Temperature sensitivity**: Does reasoning degrade at higher temperatures?

## Related Findings

Same session (L004) also demonstrated:
- Emotional engagement (joy mirroring)
- Code loop understanding (for i in range(3))
- Structured output consistency

Earlier sessions established:
- Tool syntax recognition (L002)
- Memory cue interpretation (L003)
- Mode switching awareness (L005)

Together these suggest SAGE has a richer capability floor than its size might suggest.

---
**Session Data**: `/sage/raising/sessions/latent_exploration/L004_20260126-001704.json`
**Session Log**: `private-context/autonomous-sessions/sprout-latent-L004-20260126.md`

## References

- Frederick, S. (2005). Cognitive reflection and decision making. *Journal of Economic Perspectives*, 19(4), 25-42.
