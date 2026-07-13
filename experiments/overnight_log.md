# Overnight hook-grasp goal hunt — journal

**Goal (both must hold, measured by `scripts/hook_grasp/eval_goals.py`):**
1. `catch_rate >= 0.30`  (>=30% of episodes contain a catch)
2. `hold_success_given_caught >= 0.80`  (>=80% of caught episodes reach a 100-consecutive-attached-step hold)

**Radius policy (user):** generous first (~0.10 m), tighten only if 30% is beaten.

## Machines / resources
- **ALC** (`ssh ALC`, host marvin, user diggelen, repo `~/AerialMan`, venv `.venv`): RTX 4090 24 GB.
  - User AUTHORISED killing their 3 Hebbian ES jobs → killed 2026-07-13 ~19:35. **Full 24 GB is ours.** ~5 PPO arms fit.
  - Code at `~/AerialMan` mirrors local exactly (live sync). Launch from there. Launch helper: `~/launch_wave1_alc.sh`.
- **LOCAL** (8 GB): 1 PPO arm.
- => ~5-6 parallel arms per wave. Each arm ~1.5 h to a decision point.
- Logs: ALC `~/AerialMan/logs/skrl/ppo_flycrane_hook/<rundir>` and `~/AerialMan/experiments/v22*_alc.log`; LOCAL `logs/skrl/...` + `experiments/v22*_local.log`.

## Method
- Base cfg is LEFT UNTOUCHED. Every arm = Hydra CLI overrides `env.X=Y` / `agent.agent...=Z` on `scripts/skrl/train.py`.
- COMMON goal-base overrides (all arms):
  `env.attachment.capture_radius=0.10 env.capture_radius_start=0.15 env.hold_success_steps=100 env.hold_success_steps_start=30 env.hold_curriculum_steps=150000 env.spawn_dist_mu=0.45`
- Eval: `eval_goals.py --hold_steps 100 --capture_radius 0.10 --num_envs 512 --steps 4000` on the latest checkpoint.
- Early-kill: if `attached_frac < 0.03` and flat at ~120k steps, kill and move on.
- Diagnosis: at each decision point read tensorboard (catch/approach/proximity/alive, attached_frac, terminations) before deciding next config.

## Core hypothesis
v21 collapsed into a **loiter local optimum**: drone hovers ~25 cm off wall farming proximity(2.0)+alive(3.1)/step, never commits to a catch (attached_frac ~2%, success 0). Fix = make loitering unprofitable + catching the dominant reward + easy enough motor skill to bootstrap.

---

## Experiment ladder

### Wave 1 (RUNNING). All arms share COMMON goal-base. num_envs 128 (B=96).
- **B — kill-loiter** (ALC pid 300155): `alive=0 prox=1.5 catch=30 progress=20`. Remove loiter income, boost catch/hold income.
- **C — easy motor-skill** (LOCAL pid 871334): `alive=1 prox=2 spawn_mu=0.35 radius=0.13`. Diagnostic ceiling: can an EASY catch hit 30%? If not, blocker is exploration/arch, not difficulty.
- **D — kill-loiter + easy** (ALC pid 300685): B+C combined: `alive=0 prox=1.5 catch=30 progress=20 spawn_mu=0.35 radius=0.13`.
- **E — incentive-heavy** (ALC pid 300955): `alive=1 prox=2.5 catch=40 success=120 progress=25`. Keep some living but make catch+success dominate.
- **F — commit / low-explore** (ALC pid 301228): `alive=0.5 prox=1.5 catch=30 progress=20 policy.max_log_std=-0.5`. Force the policy to sharpen (std was pinned at cap=1.0).

Decision at ~100k steps (~1h): `eval_goals.py` each; keep the 1-2 best config directions, kill the rest, seed wave 2.

**Wave-1 launched 2026-07-13 ~19:31-19:35.** Eval cadence: first goal-eval when arms hit ~100-150k (~20:20+).
**Eval how:** ALC arms — eval ON ALC: `ssh ALC 'cd ~/AerialMan && .venv/bin/python scripts/hook_grasp/eval_goals.py --headless --device cuda:0 --checkpoint logs/skrl/ppo_flycrane_hook/<run>/checkpoints/<latest>.pt --num_envs 512 --steps 4000 --hold_steps 100 --capture_radius <arm radius: B/E/F=0.10, C/D=0.13> --out /tmp/eval_<arm>.json'`. LOCAL arm C — eval locally same script. Read the printed JSON (GOALS_MET / catch_rate / hold_success_given_caught).
**Radius per arm for eval:** B=0.10, C=0.13, D=0.13, E=0.10, F=0.10.

---

## Progress log
- 2026-07-13 ~19:40: committed session state on branch `hook-grasp/overnight-goal-hunt` (799a9ad).
- 2026-07-13 ~20:35 (wave-1 first eval): **KEY FINDING** — deterministic eval >> noisy training curves. Arm **C @125k: catch_rate=0.413 (GOAL A MET), hold|caught=0.460, attached_frac=0.099**. Training showed success=0 only because exploration noise (std~1.0) knocks the hook off; the trained policy catches 41% and holds 46% deterministically.
  - ALC arms @~50k all same pattern (catch spikes, attached_frac 1-4%, success 0). B(totR-7) & E(totR-12.8) trending negative/off-target.
  - **Reframe: catching is solved-ish (>=30% reachable); the GAP is HOLDING (46%->80%, goal B).** Levers: forgiving detach model + hold-phase rewards.
  - Actions: killed B, E. Kept C(local,best), D(ALC easy twin), F(ALC commit/low-explore). Launched wave 2 (G,H) on C's easy base.
- 2026-07-13 ~21:20 (wave-2 eval): **C COLLAPSED** — re-eval @220k = 0.226/0.248 (vs 0.413/0.460 @125k). Classic peak-then-collapse. **C's BEST = 125k checkpoint (0.413 catch / 0.460 hold), saved. Killed C's process.**
  - D @90k (kill-loiter, alive=0): 0.101 catch / 0.362 hold — `alive=0` is WORSE than C's `alive=1`; removing the alive bonus hurt. So C's light-loiter base is right.
  - F (low-explore): eval compute-starved by 4 ALC training arms, pending -> read next wake.
  - **Two lessons:** (1) must eval PEAK checkpoints, not latest (collapse). (2) stability matters — likely exploration-driven; F tests it, and wave-3 J bakes in anti-collapse (low std + low entropy).
  - Launched **wave 3 J (combo)** on freed local GPU: C-base + sticky + hold-reward + stability.
- Next (~22:05): read F; eval J @~90k (local, fast); G/H still cooking (~70k). Eval arms at their PEAK (watch tensorboard attached_frac peak).

### Wave 3 (RUNNING)
- **J — combo** (LOCAL): C-base (alive=1,prox=2,spawn=0.35,radius=0.13) + STICKY(slack_dwell=15,det_v=0.5,det_h=0.1,slack_margin=0.05) + HOLD(tension=12,lock=12,progress=25) + STABILITY(max_log_std=-0.7,entropy=0.001). Hypothesis: hold levers lift 46%->higher, stability prevents the collapse that killed C.

**F EVAL (arrived ~21:35): F @100k (low-explore max_log_std=-0.5, radius 0.10) = catch 0.108 / hold 0.579 / att 0.054.**
=> **KEY: low exploration gives the BEST hold (58% vs C 46%, D 36%).** Confirms exploration noise hurts holding; calmer policy holds better. Catch low (11%) mostly due to radius 0.10 + less exploration during training. **This validates J** (easy high-catch base + even lower std -0.7 + hold levers). Winning recipe forming: easy catch base (radius 0.13, alive=1) + LOW exploration + hold support -> both goals. J is the arm to watch; if J underwhelms, try F-recipe at radius 0.13.

### Wave 2 (RUNNING) — attack the HOLD gap, on C's proven easy base (spawn=0.35, radius=0.13, alive=1, prox=2)
- **G — sticky hook** (ALC): `attachment.slack_dwell_steps=15 detach_p_vertical=0.5 detach_p_horizontal=0.1 slack_margin=0.05`. Forgiving detach so brief tension dips don't drop the hook -> policy can practise the hold.
- **H — hold-reward** (ALC): `tension_weight=12 lock_weight=12 progress_weight=25`. Strongly reward maintaining tension/lock after catch.

## Results table
| wave | arm | machine | run dir | steps@eval | catch_rate | hold\|caught | attached_frac | decision |
|------|-----|---------|---------|-----------|-----------|-------------|---------------|----------|
| 1 | B killloiter | ALC | - | 60k | - | - | 0.033pk | KILLED (totR neg, off-target) |
| 1 | C easy | LOCAL | ..._v22C_easy | **125k** | **0.413** | **0.460** | 0.099 | BEST-SO-FAR; collapsed @220k(0.23/0.25) -> KILLED, 125k ckpt saved |
| 1 | D killeasy | ALC | ..._v22D_killeasy | 90k | 0.101 | 0.362 | 0.022 | weak (alive=0 hurt); let run, low priority |
| 1 | E incentive | ALC | - | 52k | - | - | - | KILLED |
| 1 | F commit | ALC | ..._v22F_commit | 100k | 0.108 | **0.579** | 0.054 | BEST HOLD; low-explore wins for holding. radius 0.10 limits catch. Let run |
| 2 | G sticky | ALC | ..._v22G_sticky | ~40k | (cooking) | | | RUNNING |
| 2 | H holdreward | ALC | ..._v22H_holdreward | ~40k | (cooking) | | | RUNNING |
| 3 | J combo | LOCAL | ..._v22J_combo | - | | | | RUNNING (sticky+hold+stability) |
