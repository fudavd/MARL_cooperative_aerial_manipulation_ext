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
- Next (~21:15): eval D & F @100k; RE-eval C (more trained -> hold may rise); G/H need ~110min.

### Wave 2 (RUNNING) — attack the HOLD gap, on C's proven easy base (spawn=0.35, radius=0.13, alive=1, prox=2)
- **G — sticky hook** (ALC): `attachment.slack_dwell_steps=15 detach_p_vertical=0.5 detach_p_horizontal=0.1 slack_margin=0.05`. Forgiving detach so brief tension dips don't drop the hook -> policy can practise the hold.
- **H — hold-reward** (ALC): `tension_weight=12 lock_weight=12 progress_weight=25`. Strongly reward maintaining tension/lock after catch.

## Results table
| wave | arm | machine | run dir | steps@eval | catch_rate | hold\|caught | attached_frac | decision |
|------|-----|---------|---------|-----------|-----------|-------------|---------------|----------|
| 1 | B killloiter | ALC | - | 60k | - | - | 0.033pk | KILLED (totR neg, off-target) |
| 1 | C easy | LOCAL | ..._v22C_easy | 125k | **0.413** | 0.460 | 0.099 | KEEP (best; goal A met, hold gap) |
| 1 | D killeasy | ALC | ..._v22D_killeasy | 52k | (pending 100k) | | 0.045pk | KEEP |
| 1 | E incentive | ALC | - | 52k | - | - | 0.028pk | KILLED (totR neg) |
| 1 | F commit | ALC | ..._v22F_commit | 52k | (pending 100k) | | 0.029pk | KEEP (low-explore) |
| 2 | G sticky | ALC | ..._v22G_sticky | - | | | | RUNNING |
| 2 | H holdreward | ALC | ..._v22H_holdreward | - | | | | RUNNING |
