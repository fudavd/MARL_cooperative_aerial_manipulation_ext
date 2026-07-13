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

## Results table
| wave | arm | machine | run dir | steps@eval | catch_rate | hold\|caught | attached_frac | decision |
|------|-----|---------|---------|-----------|-----------|-------------|---------------|----------|
| 1 | B killloiter | ALC | (pending) | | | | | |
| 1 | C easy | LOCAL | (pending) | | | | | |
| 1 | D killeasy | ALC | (pending) | | | | | |
| 1 | E incentive | ALC | (pending) | | | | | |
| 1 | F commit | ALC | (pending) | | | | | |
