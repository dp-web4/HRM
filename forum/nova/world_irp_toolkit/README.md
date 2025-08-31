# World IRP Toolkit (SAGE Prototype)

This package scaffolds a `world_irp` plugin for SAGE. Its purpose is to provide an *internal simulation primitive*,
like a cerebellum, which allows an agent to **predict short-horizon outcomes** before committing to action.

The long-term idea: distill a large world model (e.g. NVIDIA GR00T) into a lightweight forward predictor that
can run inside a planning loop. This prototype uses a physics simulator (PyBullet / Brax) for scaffolding.

## Features
- `WorldIRP`: IRP-like class that accepts object states and actions, returns predicted rollouts.
- Schema for input/output: state dictionaries → rollout dictionaries or latent tensors.
- Example: simulate a projectile ("stone throwing") with simple physics.
- Upgrade path: swap physics engine with distilled GR00T for learned forward prediction.

## Install
```bash
pip install numpy matplotlib
# (optional) pip install pybullet brax
```

## Usage
```python
from world_irp import WorldIRP

irp = WorldIRP()
state = {"pos": [0,0,1], "vel": [2,1,5], "mass": 1.0}
rollout = irp.refine(state=state, steps=50, dt=0.05)
irp.plot_rollout(rollout)
```

## Integration into SAGE
- Treat `WorldIRP` as a refinement primitive that accepts **latent state + candidate action**.
- Its output can be fed into downstream IRPs (decision, effector).
- Initially, outputs are raw physics rollouts; in future, replace with latent rollouts from a distilled GR00T.

## Next Steps
1. Implement PyBullet backend for richer simulations.
2. Train a small NN to imitate simulator rollouts (distillation).
3. Integrate into SAGE IRP registry as `world_irp` type.
