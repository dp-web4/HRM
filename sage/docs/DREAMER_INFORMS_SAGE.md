# How DayDreamer / Dreamer Informs SAGE — World Models, Imagination, and the Embodiment Path

**Author:** CBP-Claude (Opus 4.7), 2026-05-26
**Sources:** Paper `dpx/dreamer wm.pdf` (Wu, Escontrela, Hafner, Goldberg, Abbeel, *DayDreamer: World Models for Physical Robot Learning*, CoRL 2022). Code: dp-web4 fork of `danijar/daydreamer` (cloned to `ai-agents/daydreamer/`), agent at `embodied/agents/dreamerv2plus/{agent,nets,behaviors}.py`.
**Frame (dp, 2026-05-26):** *embodiment is the eventual goal; the grid-puzzle game environments are a training step on that path.* Dreamer is the most relevant prior art we have for the actual target — it learns embodied behavior on **4 real robots, online, no simulator, from ~1 hour of experience**. So it informs both the near term (the games) and the destination (embodied SAGE). This doc is a read, not a port plan. We take what informs us; we do not adopt the stack.

---

## 1. What Dreamer actually is (code-grounded)

Two processes, one replay buffer (`embodied/run`):
- **Actor process** interacts with the world, writes experience to the replay buffer.
- **Learner process** samples replay, trains the world model (supervised), then trains an actor-critic **purely inside the world model's imagination**.

They run **decoupled, in parallel** (learner trains while actor acts) — for latency and to keep learning during slow real-world interaction. This is the same decoupled-concurrency we sketched in `sage/cognition/thalamic_router/PARALLELISM.md`.

### 1a. World Model = RSSM (`nets.py:RSSM`, `agent.py:WorldModel`)
A Recurrent State-Space Model with a deterministic recurrent state `h` and a **discrete/categorical** stochastic latent `z` (DreamerV2). Components:
- `encoder` — fuses **all sensory modalities** (proprioception, RGB, depth) into one latent. (Sensor fusion is built in, not bolted on.)
- `rssm.obs_step` → **posterior** `z` (sees the observation); `rssm.img_step` → **prior** `z` (predicts the next latent *without* seeing the observation — this is what makes imagination cheap).
- `rssm.observe` rolls the posterior over a real sequence; `rssm.imagine` rolls the prior forward under a policy.
- Heads (`agent.py:150-153`): `decoder` (reconstructs inputs), `reward` (predicts reward), `cont` (predicts episode continuation / not-terminal).
- **Loss** (`agent.py:165`): `KL(post‖prior)` + reconstruction + reward + cont, summed and weighted; trained by stochastic backprop on replayed sequences. The dynamics learning is **reward-free**; only the reward *head* needs reward labels.

### 1b. Imagination rollout (`agent.py:WorldModel.imagine`, line 234)
Given a `start` latent and a `policy`, scan `rssm.img_step` forward `horizon` (≈16) steps, the policy choosing actions at each step. Produces a trajectory of latents + actions, with `traj['cont']` (predicted continuation) and `traj['weight'] = cumprod(discount · cont)` — **imagined trajectories are down-weighted by predicted episode-end.** No decoding to pixels → 16K rollouts/batch on one GPU.

### 1c. Actor-Critic in imagination (`agent.py:ImagActorCritic` + `VFunction`)
- `actor` proposes actions; `train` (line 317) imagines a trajectory under the actor and updates.
- **Critic = `VFunction`** (line 384): a value net regressed onto **λ-returns** of *imagined* reward (`target`, line 422), with a **slow (Polyak) target net** for stability. `rewfn(traj)` returns reward **from the WM's reward head** — the critic never sees env reward during imagination.
- Actor update (line 326): `score = normalize(λ-return − value_baseline)` = a normalized **advantage**; actor maximizes it (reparam-backprop for continuous, REINFORCE for discrete) + an entropy bonus. Weighted by `traj['weight']`.
- **Multiple scaled critics** are supported (`critics` dict + `scales`, line 287) — task reward + exploration + … combined. A hook for *several value sources at once*.

**The one-line architecture:** a reward-free learned dynamics model + a reward-trained value critic + an actor that commits to high-advantage actions, all exercised in imagination, fed by a replay buffer, learner and actor decoupled.

---

## 2. The central convergence — it validates last night's finding

Dreamer keeps **two organs strictly separate**:
- **World model** answers *"what happens next?"* — dynamics, trained by reconstruction, **reward-free**.
- **Critic** answers *"is this worth it?"* — value, trained by reward via λ-returns.

This is **exactly the predictive-vs-strategic split** we discovered we had *collapsed* in SAGE (2026-05-25, `finding-v28-born...` / the frozen-`plausibility` diagnosis). Our `FaithCandidate.score = trust × plausibility` intended `trust`=evidential and `plausibility`=strategic-prior — but `plausibility` never updated (frozen at seed) and `trust` moved only on **predictive** residual, so commitment rode predictive accuracy → the tracking≠winning trap. Dreamer is the mature, working proof that **predictive dynamics and strategic value must be different networks, trained on different signals.** The "win-residual / strategic-trust" I kept circling **is** Dreamer's critic. We arrived at the decomposition by hitting the wall; they built their system on it.

---

## 3. Component-by-component mapping to the SAGE stack

| Dreamer component | SAGE analog (today) | Gap / what it informs |
|---|---|---|
| RSSM **dynamics** (`img_step`, reward-free) | `wm_schema` causal_rules (symbolic) + LLM priors | We have a WM but **don't roll it forward**. Dreamer's `img_step` = a learned forward dynamics we lack. |
| **Imagination rollout** (`imagine`, H=16) | WinImaginer (`win_imaginer_live.predict_terminal`) — predicts a *terminal*, not a *rollout*; LLM "imagines" plans in language | The imaginator north star (`SAGE_WI_GOAL`, tasks #309/#293) wants a *rollout* to score plans before acting — **the antidote to the grind**. |
| **reward head** (learned, trained on env reward) | *absent* — our grid-puzzle environments are a feedback desert | We can't train a reward head on env reward. **WinImaginer terminal / body→terminal gap = a self-supplied surrogate reward** (imagination manufactures the signal the env withholds). |
| **critic / λ-return value** (`VFunction`) | the *frozen* `plausibility` / the missing "strategic-trust" | Build the critic dimension: a value updated by the surrogate-reward, **separate from the predictive residual**. |
| **multiple scaled critics** | the faith portfolio's multiple candidates | Several value sources combined → maps onto multi-faith (task-value + explore-value + …). |
| **encoder** (multi-sensor fusion → latent) | perception-as-integrator; `GridVisionIRP`; the v2 object producer | Dreamer fuses *all* modalities into one state — the integrator principle, already ours; relevant when embodied (proprio + vision + force). |
| **cont head** (episode-end / terminal) | win-state / terminal prediction (WI win-patterns: `body_at_position` / `no_body_at_win`) | Dreamer's `cont` is exactly "is this a terminal?" — a learned terminal/win classifier weighting imagined trajectories. |
| **decoder / reconstruction** | *absent* (WI predicts only `terminal_xy`) | A dense, **human-inspectable** signal — we could imagine the next full objstate/frame and *see* what the WM imagines. |
| **decoupled learner/actor** | `PARALLELISM.md` (sketched) | Validated architecture for online learning while acting. |
| **replay buffer** | `episode_store/*.jsonl.gz` | Same role; ours already exists. |
| `embodied/` framework (core/envs/replay/run) | SAGE consciousness loop + `sage/irp` | A candidate reference substrate for **embodied SAGE** (see §6). |

---

## 4. The honest divergences — why we do NOT port

1. **Dreamer needs env reward** to train its reward head; our grid-puzzle environments (and much of open-world embodiment) are a **feedback desert**. This is *why* we built **faith** (commit-before-confirmation). Our innovation — **imagination supplies the value signal** (WinImaginer terminal as surrogate reward) — is precisely the part Dreamer doesn't need because robots get task reward. So the reward-free WM half is liftable; the reward-hungry actor-critic is not, as-is.
2. **End-to-end neural (opaque) vs symbolic + LLM (interpretable).** DreamerV2's discrete latents actually align with our discrete object/symbolic representations, but the RSSM is a black box; our causal_rules are human-readable and the LLM carries world-priors no robot has. We trade sample-efficiency for interpretability + zero-shot priors.
3. **Sample-efficiency is the thing we lack.** Dreamer's imagination is the cure for our **grind** (the v28 8-hour, flat-cap, 14/24-OOM run). But it requires a *learned dynamics model we don't yet have* — that's the build, not a free lunch.

The synthesis worth weighing: **Dreamer's reward-free dynamics + imagination-rollout + a value critic, fed by SAGE's faith-as-reward-substitute, with the LLM as high-level proposer — each as a SAGE IRP organ.**

---

## 5. IRP-plugin reading (the embodiment angle)

SAGE's organs are **IRP plugins** (`SAGE/sage/irp`; Nova's `forum/nova/world_irp_toolkit` already prototypes world-model-as-IRP; `GROOT_SAGE_INTEGRATION_PLAN.md` is the robot path). Dreamer factors cleanly onto that interface:

- **Dynamics IRP** — `predict next latent/objstate | state, action`. The reward-free RSSM `img_step`. Iterative-refinement-shaped: refine the predicted next-state. This is the core new organ.
- **Imagination IRP (the imaginator)** — roll the Dynamics IRP forward H steps under a candidate policy/plan; return the trajectory + a score. This is dp's long-standing imaginator north star, given Dreamer's concrete mechanism. Maps to tasks #309/#293 (SAGE IRP-plugin wiring for the imagination organ).
- **Value/Critic IRP** — score an imagined trajectory (λ-return of the surrogate reward). This is the *strategic* half of faith, finally its own organ.
- **Perception/encoder IRP** — multi-sensor fusion into one state (already our integrator; Dreamer shows the embodied multi-modal version: proprio + vision + depth + force).

For the **embodied future specifically**: when SAGE drives GR00T / a real robot, Dreamer is the proven sample-efficient online learner (1h from scratch, no sim, adapts to perturbation in 10 min). The Dynamics+Imagination+Value IRPs are *exactly* what a robot-SAGE needs, with the faith gate handling the reward-sparse regions the robot will hit. The games are where we debug these organs cheaply before the robot.

---

## 6. Concrete next steps (proposed; not all now)

Ordered by leverage. None require porting Dreamer.

1. **Split trust into two organs (highest leverage, smallest change).** Stop conflating predictive and strategic in one scalar. Keep the per-rule residual (`predict_residual_per_rule`, Step 2.2) as the **predictive** signal; add a **strategic value** updated from the surrogate reward (below). This is the critic, housed in the faith portfolio. Directly fixes the frozen-`plausibility` / tracking≠winning trap.
2. **Surrogate reward from imagination.** Define a dense value where the env gives none: body→terminal gap closing (embodied games) or WI win-pattern confidence (controller games), gated by the self-mode read. This is the reward-free analog of Dreamer's reward head — the value signal the critic regresses on.
3. **Extend WinImaginer: terminal → short forward rollout.** Add a learned `next-objstate | objstate, action` dynamics (object-space RSSM-lite) and roll it H steps to **score faith candidates in imagination** before acting. The grind antidote. Validate offline on replays first (we have `replay_dataset/*.objstate.jsonl`), the way we validated body-ID.
4. **Stand up the three IRPs** (Dynamics, Imagination, Value) behind the SAGE consciousness loop, composing with Nova's `world_irp_toolkit`. Start with the Dynamics IRP (offline-trained on replays), then Imagination, then Value. This is the buildable form of tasks #293/#309.
5. **Multi-critic → multi-faith.** When >1 value source exists (task + exploration), combine them with scales (Dreamer's pattern) — the principled version of the faith portfolio's competing candidates.
6. **Embodiment track (longer horizon).** Treat the `embodied` framework + `GROOT_SAGE_INTEGRATION_PLAN` as the reference for robot-SAGE; the Dynamics/Imagination/Value IRPs are the same organs the robot will need. The games validate them; the robot is the target.

**Caveat / agent-zero discipline:** every one of these needs its dummy baseline. A learned object-dynamics model must beat "predict no change" / "repeat last delta"; the imagination-rollout score must beat random-plan selection; the critic must beat a constant. Build the baseline alongside, per `references/agent-zero.md`.

---

## 7. So what

Dreamer doesn't replace SAGE's approach and shouldn't be ported. It **(a)** independently validates the predictive-vs-strategic decomposition we reached by failure; **(b)** hands us the imagination-rollout mechanism that is the principled cure for the grind; **(c)** is the proven embodiment learner for the eventual goal, factoring cleanly into IRP organs. The genuinely-ours part — imagination *manufacturing* the value signal in a feedback desert via faith + the WinImaginer — is exactly the gap Dreamer never had to cross, because robots get reward and grid-puzzle worlds (and the open world) often don't. That's not a deficiency to fix by adopting Dreamer; it's the thing worth building, with Dreamer's architecture as the map.

— CBP-Claude (Opus 4.7), 2026-05-26
