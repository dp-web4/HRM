
# SAGE ↔ Totality Adapter Spec (v0.1)

This spec defines the interface between **SAGE** and a **Totality**-style cognitive world model so the latter can act as a **Cognitive Sensor** inside SAGE while remaining independently evolvable.

---

## 1) High-Level Diagram

```
Physical Sensors ─┐
Memory Sensor  ───┼──► L-Module ──► H-Module ──► Strategy / Act
Totality Adapter ─┘          ▲             │
     (Cognitive Sensor)      └──── Trust ◄─┘
```

- **Totality Adapter** exposes *read / write / imagine / activate* ops over the Totality store.
- **SAGE** treats the adapter as a cognitive sensor whose outputs receive a **trust score** and can be selected or fused at H-Module.
- Affect (valence/arousal/coping) maps to **SNARC** gates for selective write / sampling / priority.

(…full details omitted for brevity…)
