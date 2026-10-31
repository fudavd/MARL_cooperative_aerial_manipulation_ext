# D28 proposal: bound the policy's commands on the corrected drone (d16v)

Drafted 2026-10-31 (times from `date`). Read-only and local: no code is written, synced or submitted. The user decides.

## Why

d16u / d16u2 is d16t's recipe on the corrected 0.617 kg drone (T/W 2.76). Its eval failures rise with training.

| row (nocoll) | fly_low | blowup | episodes |
|---|---|---|---|
| 10k | 2 | 1 | 128 |
| 12k | 31 | 2 | 141 |
| 14k (resumed) | 64 | 9 | 161 |
| 16k (resumed) | 14 | 3 | 129 |
| 18k (resumed; nocoll and joint-share rows identical) | 968 | 53 | 1036 (median episode 3.18 s; 75 catches from ~8x the episodes, not an onset) |
| d16r 18k (heavy drone) | 10 | 1 | 129 (58 catches) |
| d16r 14k (heavy drone) | 7 | 0 | – |

The D26 rollout probe on d16u it_14000 (deterministic policy, tracker eval cfg) found the following.

- All 64 fly_low and all 9 blowups are FREE drones that never caught.
  - The terminations come 1.84 s (median) after spawn.
  - The drone falls from 1.73 m at −5.6 m/s.
  - Body rate |ω| peaks at 38 rad/s.
  - No case is attached-pulled-down.
- Surviving episodes fly at normal altitude. The per-drone mean z is 1.81 m (d16r: 1.85 m).
- The policy commands these rates itself. Over the last 1 s, the mean |body-rate action| is 14 rad/s, against d16r's 6.
- Actions reach the controller unclipped:
  - construction.py `_pre_physics_step` sets `sp["lin_acc"] = act[:, :3]` and `sp["body_rates"] = act[:, 3:5]`, with no clamp.
  - geometric.py:220 tracks the rate setpoint through a P loop (kp_rate 25), with no rate limit.
- The heavy drone could not realise these commands. The corrected drone does.

Pass 2 of the probe gives command and realised-rate distributions per drone-step, labelled by episode outcome. Files: `~/flycrane_tracker/probe/cmd_{u14_light,r48_heavy}.json`.

| population | rate cmd \|·\| q50 / q90 / q99 (rad/s) | realised \|ω\| q50 / q90 / q99 | \|lin_acc cmd\| q50 / q90 / q99 (m/s²) |
|---|---|---|---|
| d16u 14k, survivors, free | 4.9 / 7.9 / 10.6 | 5.9 / 7.8 / 10.3 | 8.4 / 11.6 / 13.8 |
| d16u 14k, survivors, attached | 4.5 / 5.9 / 6.5 | 6.1 / 6.6 / 6.8 | 9.5 / 13.3 / 13.4 |
| d16u 14k, failing episodes, free | 6.2 / 22.4 / 46.0 | 6.7 / 20.4 / 35.4 | 9.3 / 23.8 / 46.8 |
| d16r 48k (heavy), survivors, free | 4.8 / 10.5 / 17.1 | 4.0 / 8.2 / 11.3 | 22 / 39 / 54 (unrealisable) |

The last 1 s before each catch, for d16r 48k (194 catches):

| quantity | q50 | q90 | q99 |
|---|---|---|---|
| peak \|ω\| realised (rad/s) | 9.3 | 11.6 | 13.2 |
| peak rate command (rad/s) | 12.1 | 17.2 | 21.3 |
| mean rate command (rad/s) | 5.3 | 6.8 | 8.0 |

d16u 14k had only 2 catches: peak |ω| 26-32 rad/s and peak lin_acc 16 m/s². That sample is too small to use.

## Lever 1 (primary): action bounds, env-side

**What:** clamp per axis in `_pre_physics_step`, before the setpoints are built:
- body rates `act[:, 3:5]` to ±B_ω;
- lin_acc `act[:, :3]` to a vector norm of at most 15 m/s².

**Where:** apply the clamp to `act` before `self.actions[:, i] = act`, so the stored actions are bounded too. `self.actions` feeds only the reward terms: action_rate, action_accel, action_smooth and body_rate (lines 754-761). It does not feed the observations, so the smoothness terms will see the bounded command, as wanted. The 5-action contract stays unchanged.

**Interaction with PPO:** rsl_rl stores the unclipped sample for the log-probability. This is the standard "clip in env" pattern. Outside the bound, the gradient on the mean is the PPO term only. Expect the std on those dims to stop being pushed by the plant response.

**B_ω (rad/s per axis), recommended 8:**
- Per-axis ±8 allows |ω_xy| up to 11.3. That covers what d16r's catches actually realised (peak q90 11.6) and the d16u survivors' commands (q99 10.6).
- It cuts the failing tail (q90 22, q99 46).
- ±6, the PM's first number, allows |ω_xy| up to 8.5. That is below the survivors' q90-q99 command range and below the median peak a catch realised (9.3). It is the stricter alternative if ±8 still tumbles.

**lin_acc (norm ≤ 15 m/s²):** the corrected drone's maximum is about 25 N / 0.92 kg − g ≈ 17 m/s² net up. The survivors command at most 13.8 (q99). The failing tail reaches 24-47. d16r's 22-54 were unrealisable anyway.

**Does the lin_acc bound matter physically, or only for the learner?** Both, a little.
- Rotor thrust does saturate. geometric.py's clamps (226/235) are commented out, but indi.py clamps the collective to [0, 25] N (line 122) and each rotor to [0, 6.25] N (line 136).
- The motor model allows up to 2800 rad/s, which is 12.25 N (thrust_map 1.5625e-6). So the INDI clamp binds first.
- A command beyond the plant's ≈17 m/s² cannot be realised. Worse, per-rotor clipping removes torque authority while the clamp is active, so a saturating lin_acc command can cost attitude control. That is a second route to tumbling.
- So the bound removes a physically harmful regime and also the learner's incentive to sit in it.
- On the heavy drone almost every d16r command saturated, which is likely why its rate commands were harmless there.

**Regression checks with the bounds on:**
- The scripted actions are inside the bounds. `hold_actions` sends body rates 0 and lin_acc clamped to ±6 per axis (norm ≤ 10.4, much less once settled).
- The vbox ramp adds at most 8 m/s² along (0.7 n + 0.7 up). In the last runs, risk and real motion fired at ramp levels ≤ 4.3, well inside a 15 norm. Levels above ~7 may be clipped near the end of the ramp, which does not change the reported first-risk / first-motion levels.
- No check is expected to need the bound lifted. This must be verified by running all four checks with the bounds on (debug qos, bands as 3235054) before any launch.
- Proposed addition: a regression `--check bounds`. Saturate the actions with random ±50 commands and assert that the realised setpoints and `self.actions` are inside the bounds.

## Lever 2 (fallback only): a body-rate penalty that bites

- d16u runs `w_body_rate=1.0` (the cfg default is 0.0).
- construction.py:761 is a saturating BONUS, `w·dt·exp(−|a_rate|)`. Its gradient is about −exp(−|a|), which is 8e-7 at 14 rad/s and 2.5e-3 at 6 rad/s. It cannot pull back a tumbling command.
- Fallback, only if bounded runs still tumble inside ±B_ω: a quadratic penalty on the REALISED body rate, `−w·dt·|ω_xy|²`, or on the command beyond a free band, `−w·dt·relu(|a_rate| − 4)²`. Not in d16v.

## Lever 3 (listed; PM recommends against for now): swing_assist floor

- The spawn kick fades from 2.2 to 0 over 0-300k control steps (it 12.5k). fly_low rises over the same span.
- A floor (for example 0.5) would keep the policy from making the swing itself.
- It changes the task, is unverified as the cause, and d16r faded it without this failure. Not proposed now.

## Arm plan (protocol v1.4)

- **d16v** = d16u's recipe + Lever 1 (B_ω = 8, lin_acc norm ≤ 15).
  - FROM SCRATCH (user direction 2026-10-22; d16u's policy has already learned the rates).
  - Seed 42, 60k it, 512 envs, `phantom_link_mass=0.001`.
  - The guards from 65a865f/14769e2 are on (`--nonfinite_guard`, obs guard).
  - The sbatch propagates the exit code (`rc=$?; ...; exit $rc`).
- **Code:** a cfg flag (`action_bound_rate`, `action_bound_acc`, default off = unbounded), local commit, PM diff, regression with bounds (plus `--check bounds`), then the user's go.
- **Pre-registered checks** (d16u's (1)-(4) plus one):
  1. onset = first row with ≥50 catches, vs d16r's 20k-24k;
  2. held ≥ 0.8, detaches as d16r;
  3. first shared-load 1 s hold by 40k;
  4. 4k plant sanity: (fly_low + blowup + oob)/episodes ≤ max(2× d16r's 4k rate, 0.12) and median episode ≥ 30 s;
  5. NEW: (fly_low + blowup)/episodes at 12k and at 14k < 2× d16r's 14k rate (2 × 7/128 ≈ 0.11, so < 14/128).
- **Eval:** the tracker rows as for d16u, with the same eval cfg plus the bound flags. An evaluation policy must see the same plant as in training.

## d16u2 meanwhile

| option | for | cost |
|---|---|---|
| keep as unbounded reference (PM leans this way) | the matched comparison for d16v's check (5) at every 2k, and it shows whether the unbounded policy recovers by itself | ~31.5 h more on izar (17.9k at 01:02; 1335 it/h; 42k to go). The laptop tracker would carry two runs: 3 rows per 2k each, ~40-45 min of eval per 2k per run, against ~1.5 h per 2k of training. That roughly saturates the one-GPU tracker. Proposed: d16u2 drops to the nocoll row only (1 row per 2k) |
| stop it to free the slot | frees one izar GPU and the tracker | loses the unbounded reference beyond 18k. d16u's 10k-18k rows already document the failure mode |

**PM position after the 18k row (2026-10-31 02:07): stop d16u2 at the d16v launch.** Its deterministic policy dives in 968 of 1036 joint-share episodes within ~3 s while the stochastic training rollouts look calm; a reference that cannot survive 4 s has no comparison value beyond what is recorded. The user decides.


## PM review (2026-10-31 01:46, -c5)

Drafted by the hooking session (-6d) from the D26 probes; reviewed by the PM. Verdict: **recommend to the user as written, with
three additions.**

1. **B_omega = 8 rad/s per axis accepted** over the PM's first 6: the catches d16r actually made realised peak |omega| q50 / q90 of
   9.3 / 11.6 rad/s and the d16u survivors command q99 10.6, both inside the 11.3 that +-8 per axis allows; +-6 would cut them.
   +-6 stays the stricter fallback if d16v still tumbles inside the bound.
2. **Add a saturation metric and a sixth check.** Env-side clipping with a Gaussian policy lets the mean drift far outside the bound
   (no plant gradient there) and the policy turn bang-bang. Log the fraction of drone-steps with a saturated rate or acceleration
   command (Metrics/action_sat_rate, Metrics/action_sat_acc). Check (6): saturated fraction < 0.3 at 10k; above it, Lever 2 in its
   relu(|a| - 4)^2 form is the pre-agreed next step rather than a new proposal.
3. **d16u2: stop it when d16v launches** rather than keep it. The failure mode is characterised (10k-18k rows + the two probes), the
   matched comparison for check (5) is d16r's and d16u's existing rows, and a third GPU job (evo holds one, hooking two) risks the
   per-user GPU quota queueing d16v behind d16u2. If the queue shows a free GPU at launch time, keeping d16u2 at the nocoll row only
   is acceptable; the user decides.

Everything else stands: Lever 1 primary, Lever 2 fallback only, Lever 3 not now, d16v from scratch under protocol v1.4 with the
d16u recipe + bounds + guards + exit-code propagation, regression with the bounds on (plus `--check bounds`) before any launch, the
eval cfg carrying the same bound flags as training. Code is written as a local flycrane-meta commit for the PM diff; the regression
runs on debug qos under the hooking session's standing permission; the training launch needs the user's go in the hooking session.
