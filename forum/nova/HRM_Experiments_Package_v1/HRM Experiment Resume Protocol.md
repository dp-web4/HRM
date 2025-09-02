awesome — here’s a **clean, repeatable resume protocol** you can drop into your run docs. It keeps experiments comparable, minimizes wall-clock waste, and handles the edge cases (optimizer swaps, SWA/EMA, small arch edits).

# HRM Experiment Resume Protocol

## 0) Which checkpoint to resume from?

**Default:** the **latest \*stable\* step checkpoint** from the same recipe family

- Use: `checkpoints/hrm_arc_step_<N>.pt` with the highest `<N>` that completed a full forward+backward+optimizer step and (ideally) a recent fast-eval.
- Keep the **best** checkpoint (`*_best.pt`) for archival/baselines, but resume from the **latest** to preserve optimizer momentum and scheduler phase.

**Exceptions:**

- **Optimizer swap experiments** (e.g., AdamW→Lion): resume **model weights only** from latest step; **re-init optimizer & scheduler** (details below).
- **Architecture-touching experiments** (e.g., add DropPath, aux head): resume **model weights partially** (`strict=False`) and **re-init new params**; try to preserve optimizer where possible.

------

## 1) What to load/keep vs. reset

| Component                        | Default Resume                              | When to Reset                                                |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Model weights**                | Load from chosen checkpoint                 | Only if from-scratch control                                 |
| **Optimizer state**              | Load                                        | Reset for optimizer-swap, or if LR schedule fundamentally changes |
| **LR scheduler**                 | Load                                        | Reset if new schedule (e.g., add cyclical tail), or if you change `max_steps`/warmup logic |
| **AMP scaler**                   | Load if present                             | Reset if CUDA/driver/PyTorch version changed                 |
| **EMA/SWA weights**              | If present and continuing same method, load | If newly enabling EMA/SWA, start accumulating fresh          |
| **Global step / epoch counters** | Load & continue                             | Reset only for from-scratch control runs                     |

------

## 2) Allowed changes when resuming (safe to do)

- **Batch size** (and **grad accumulation**) — OK; step/optimizer states remain valid.
- **Validation cadence** (fast/2k, full/10k) — OK.
- **Weight decay, grad clip** — small tweaks OK.
- **DropPath (H-loop)** — add new param(s): load with `strict=False`, init new tensors, keep old ones.
- **Aux heads** (rule descriptor): same approach as above.
- **SWA tail** — enable later; keep base optimizer states.
- **Gradient noise σ schedule** — OK to add.

------

## 3) Run lineage & hygiene (make results comparable)

- **run_id:** `EXP-XX_<baseRunId>_resume@<step>_<yyyymmdd-hhmm>`
- **status.json (heartbeat, 60s):** write `{run_id, global_step, steps_per_sec, lr, last_fast_eval_step, last_full_eval_step, best_eval_acc, eta_*}`
- **lineage tag:** include `parent_run_id`, `parent_step` in your logs & artifact metadata.
- **artifacts on new best:** snapshot `weights.pt`, `config.yaml/json`, `status.json`, and a **100–200 line log window** around the best.

------

## 4) Sanity checks right after resume (1–2 minutes)

1. Print: `Loaded checkpoint @step=<N>, epoch=<E>, best_val=<x.xx>, lr=<...>`
2. Do a **dry batch** (no backprop) to confirm shapes/device/AMP are good.
3. Confirm **scheduler phase** (warmup vs cosine/tail) matches expectation.
4. Confirm **throughput** is in the expected band (±10–15%).
5. Run a **fast-eval** on a small fixed subset (1–2%) to ensure no silent corruption.

------

## 5) Suggested directory structure

```
runs/
  EXP-03_droppath_base/
    checkpoints/ hrm_arc_step_40000.pt, ... hrm_arc_best.pt
    logs/ train.log
    status.json
  EXP-03_droppath_base_resume@40000/
    checkpoints/ ...
    logs/ train.log
    status.json
```

Keep each resume in its own folder with symlink to dataset and a **copy** (or hash) of the config actually used.

------

## 6) Minimal pseudocode (robust resume)

```
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# 1) Model: allow partial load if arch changed
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
if missing: print(f"[warn] missing params: {len(missing)} (new layers?)")
if unexpected: print(f"[warn] unexpected params in ckpt: {len(unexpected)}")

# 2) Optimizer & scheduler policy
if EXP_CONFIG["optimizer_swap_at"] and RESUMED_STEP >= EXP_CONFIG["optimizer_swap_at"]:
    optimizer = make_optimizer(model.parameters(), kind="lion")
    scheduler = make_scheduler(optimizer, EXP_CONFIG, reset=True)
else:
    optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
    scheduler.load_state_dict(ckpt.get("scheduler_state_dict", {}))

# 3) Mixed precision scaler
if "scaler_state_dict" in ckpt:
    scaler.load_state_dict(ckpt["scaler_state_dict"])

# 4) Counters & book-keeping
global_step = ckpt.get("global_step", 0)
start_epoch = ckpt.get("epoch", 0)
best_val = ckpt.get("best_val_loss", float("inf"))

print(f"[resume] step={global_step} epoch={start_epoch} best_val={best_val:.4f}")

# 5) New params init (if we added DropPath/aux heads)
for n, p in model.named_parameters():
    if n in missing:
        torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))

# 6) Quick dry forward on a fixed mini-batch
with torch.cuda.amp.autocast():
    _ = model(sample_inputs.to(device))
```

------

## 7) Optimizer swap (AdamW → Lion) specifics

- **Load:** **model weights only**, **do not** load old optimizer/scheduler states.
- **LR:** start at **80%** of prior LR for the first 1–2k steps, then restore; or use a short **warm restart**.
- **WD:** keep at 0.01 initially; reduce if you see underfitting.
- **Log** the swap event in `status.json` (`"optimizer":"lion", "swap_from_step": <N>`).

------

## 8) SWA tail specifics

- Start SWA collection at a **fixed step (e.g., 50k)** or **when fast-eval improves after a stall**.
- Maintain **base weights** + **SWA weights** in parallel; evaluate both on full-val, pick the better for “best.”
- Keep SWA LR **small cyclical** (triangular, tiny amplitude) so you explore a flat basin.

------

## 9) EMA (if you enable it)

- If adding EMA mid-run: initialize EMA weights from the **current model** at resume step.
- Evaluate **both** raw & EMA; many times EMA is the early “+1%” win.

------

## 10) Repro & comparability checklist

- Fix `seed`, log torch/cuda versions, log batch/grad-accum/eval cadence explicitly.
- Add a `config_hash` (SHA256 of your config file) to every log line.
- On each eval, log **family-wise metrics** (if available) to see *what* moved.

------

## 11) Quick shell template

```
RUN_ID="EXP-03_droppath_base_resume@40000_$(date +%Y%m%d-%H%M)"
PYTHONUNBUFFERED=1 python -u train_arc_full_nova.py \
  --run_id "$RUN_ID" \
  --resume "runs/EXP-03_droppath_base/checkpoints/hrm_arc_step_40000.pt" \
  --droppath_h "0..0.2" \
  --fast_eval_every 2000 \
  --full_eval_every 10000 \
  --gate_full_eval_on_fast_improvement true \
  --write_status_json_every_sec 60 \
  --status_json_path "runs/$RUN_ID/status.json" \
  --logdir "runs/$RUN_ID/logs" \
  2>&1 | tee "runs/$RUN_ID/logs/train.log"
```

------

## 12) Abort & recover rules

- **Abort** if: loss becomes NaN/Inf, steps/sec drops >50% for >10 min, VRAM OOM loops, or eval crashes.
- **Recovery:** backtrack to the **last good checkpoint**, **reduce LR 20%**, and **replay** the last 1–2k steps. Log the recovery event.

------

### TL;DR

- Resume from the **latest stable step**; keep optimizer/scheduler unless you **intentionally** change them.
- Standardize **run lineage** and **status.json** so step/ETA/best/eval are unambiguous.
- For **optimizer swaps/arch edits**, load **weights only**, re-init the changed parts cleanly, and document the divergence.

If you want, I can tailor this into a tiny `resume.py` helper that wraps the policy (choose checkpoint, apply rules, start run) so you don’t have to remember switches each time.

##### You said:

great!  let's do a downloadable py

###### ChatGPT said:

here you go — a tidy, reusable **resume helper** you can drop into the repo:

📜 Download resume_helper.py

what it gives you:

- `load_resume_bundle(...)` — standardized loading of model/optimizer/scheduler/scaler with policies (`load/reset/swap`) and partial-weight support.
- `make_run_id(...)` — consistent run lineage (includes parent + resume step).
- `heartbeat_writer(...)` + `make_status_payload(...)` — write `status.json` every minute with step/ETA/best/eval, etc.
- CLI dry-run: `python resume_helper.py --ckpt_dir checkpoints --dry-run` to sanity-check selection without Torch code.

wire-in snippet (in your train script):

```
from resume_helper import load_resume_bundle, heartbeat_writer, make_run_id

run_id = make_run_id(args.exp_name, parent_run_id=args.parent_run_id, resume_step=args.resume_step)
bundle = load_resume_bundle(
    model, optimizer, scheduler, scaler,
    ckpt_path=args.resume,
    allow_partial_weights=args.allow_partial,
    optimizer_policy=args.optimizer_policy,  # "load"|"reset"|"swap_to_lion"
    schedule_policy=args.schedule_policy,    # "load"|"reset"
    log_fn=print,
)
write_status = heartbeat_writer(args.status_json_path, lambda: current_payload_dict(), 60)
```

want me to also generate a tiny `run.sh` template that uses this helper and sets the folder layout + log tee-ing? 

##### You said:

sure!

###### ChatGPT said:

here’s a ready-to-use launcher script:

🚀 Download run.sh

what it does:

- auto-selects latest checkpoint from `runs/<exp_name>/checkpoints` if `--resume` isn’t given
- builds a fresh log dir per run (`runs/<exp_name>_<timestamp>`)
- pipes stdout/err to `train.log` while preserving color/logging in terminal
- refreshes `status.json` via your training loop (hooked to `heartbeat_writer`)

you’ll just need to make sure your `train_arc_full_nova.py` is calling into `resume_helper.load_resume_bundle` as we outlined.

want me to also include a **minimal config.yaml template** so each run bundles its hyperparams (batch size, eval cadence, optimizer, etc.) with the logdir? 

of course

###### ChatGPT said:

here you go — a clean baseline config you can copy per run:

📑 Download config.yaml

contents:

- metadata (exp name, lineage, notes)
- dataset + model spec (seq_len, layers, hidden size)
- training loop params (batch size, grad accum, lr, warmup, wd, clip)
- validation cadence & checkpointing
- resume policies (load/reset/swap)
- paths (status.json, checkpoints, logs)

this way every run has a **self-documenting config** you can archive alongside `train.log` and checkpoints.

would you like me to also draft a **tiny python helper** that ingests `config.yaml`, merges CLI overrides, and dumps the resolved config into the run logdir at startup (so every run is reproducible)? 

##### You said:

that would complete the package, yes

###### ChatGPT said:

Thought for 9s

wrapped it up. you now have a small utility to load a YAML config, apply dotted CLI overrides, and write a **resolved** config (YAML + JSON) with a stable hash into the run logdir:

🧩 Download config_resolver.py

how to use (standalone):

```
python config_resolver.py \
  --config config.yaml \
  --logdir runs/EXP-01_$(date +%Y%m%d-%H%M) \
  --set training.batch_size=8 \
  --set grad_accum_steps=5 \
  --set eval_frequency=2000
```

what it does:

- reads `config.yaml` (uses PyYAML if available, otherwise a simple fallback),
- deep-merges CLI overrides (dot-keys supported),
- writes `resolved_config.yaml/json` to your `--logdir`,
- prints a `config hash` you can log with the run.

if you want, I can stitch `run.sh` to call this resolver before launching the trainer so every run auto-snapshots the exact hyperparams it used. 