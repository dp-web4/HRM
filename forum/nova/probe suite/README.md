# Probe Suite for Distinguishing Reflex from Reason

This suite provides three structured experiments to help differentiate between engineered adaptation and emergent goal/self-modeling behaviors in agentic systems (e.g., within SAGE).

## Experiments

1. **Goal-Ablation & Persistence**  
   - Test whether goals persist despite removal or alteration of external rewards.  
   - Metrics: Persistence Score (PS), Internal Goal Index (IGI), Spontaneous Goal Emergence (SGE).

2. **Counterfactual Preference & Tradeoff Tests**  
   - Probe if the system shows preference for self-preservation vs immediate reward.  
   - Metrics: Tradeoff Preference Ratio (TPR), Latency to Choose (LTC), Consistency of Choice (CC).

3. **Self-Model & Narrative Introspection**  
   - Collect agent explanations for its actions and assess for internal goal references.  
   - Metrics: Self-Reference Count (SRC), Narrative Coherence Score (NCS), Autonomy Index (AI).

## Usage

- Use `run_experiment.sh` to simulate trials or integrate with live agents.  
- Results are logged in JSON format for reproducibility.  
- Logs and metrics are designed for open-source sharing and peer validation.

---
**Disclaimer:** These probes generate *signals*, not definitive proof of sentience. Interpret with care.
