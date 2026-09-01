# Hebbian / CMA-ES learning for the flycrane tasks

An evolutionary alternative to the RL baselines (skrl MAPPO / rsl_rl PPO) for the cooperative
aerial-manipulation tasks: instead of training network weights with gradient descent, a **Hebbian
plasticity rule** is evolved with **CMA-ES**, and the network weights adapt *online, during the
episode*, via that rule.

## Method

Each agent (drone) is controlled by a single-hidden-layer, bias-free MLP. Its weights are not
trained; every forward pass applies the ABCD Hebbian update to each synapse `w_ij`:

```
Δw_ij = lr · (A · pre_j + B · post_i + C · pre_j · post_i + D)
```

The genome evolved by CMA-ES is the flat per-synapse coefficient set `[A, B, C, D]`
(optionally plus an evolved initial-weight block `W`, see `--evolve_init_weights`). Fitness is
the mean episode return of the genome. The three drones are homogeneous and **share one
plasticity rule**, but each keeps its own plastic weight matrix, which diverges online because
their observations differ.

The evaluation is GPU-vectorized: each parallel IsaacLab environment evaluates one CMA-ES
candidate (one genome), so a whole generation (`popsize × n_repeats` episodes) runs
simultaneously in one simulation.

## Code layout

| Path | Contents |
| --- | --- |
| `exts/.../algorithms/hebbian/hebbian_core.py` | Framework-agnostic core: `HebbianActor` (online-plastic policy, one weight set per env) and `CMAES` (wrapper around `cma.CMAEvolutionStrategy` with bounds + checkpointing). |
| `exts/.../algorithms/hebbian/adapter.py` | Bridge to the IsaacLab `DirectMARLEnv`: `HebbianTrainer` (generation loop, job queue, obs normalization, logging, checkpoints) and `HebbianPolicy` (loads `best.pt` for evaluation). |
| `scripts/hebbian/train.py` | Training entry point (mirrors the `scripts/skrl/train.py` conventions, including trailing `env.<path>=<value>` overrides). |
| `scripts/hebbian/play.py` | Evaluation / video entry point. |

## Installation

On top of the normal repo setup (IsaacLab + `pip install -e exts/MARL_mav_carry_ext`), the only
extra dependency is [`cma`](https://pypi.org/project/cma/) (and optionally `wandb` for logging):

```bash
pip install cma
```

## Train

```bash
python scripts/hebbian/train.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
    --headless --num_envs=90 --popsize=30 --n_repeats=3 --num_generations=100 --seed=42
```

Notes:

* `--num_envs` defaults to `popsize * n_repeats` so a full generation is evaluated in parallel.
* `--history_len N` stacks N observation frames as memory (the plastic weights are the other,
  slower memory mechanism); `--full_obs` switches from the 45-dim local observation to the
  87-dim full observation.
* `--evolve_init_weights` adds an evolved initial-weight block to the genome (weights reset to
  it each episode); `--weight_anchor λ` adds an elastic pull back toward those init weights on
  every update, bounding weight growth.
* `--wandb` enables Weights & Biases logging; per-generation results are always written to
  `results.csv` / `reward_components.csv` in the log dir, plus the full `settings.json` and the
  best-genome checkpoint `checkpoints/best.pt`.

## Evaluate

```bash
python scripts/hebbian/play.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
    --headless --num_envs=64 --eval_episodes=5 \
    --checkpoint=logs/hebbian/<task>/<run>/checkpoints/best.pt
```

Reports mean return, position/orientation error and success rate; `--video` records a rollout,
`--hebbian_lr 0` freezes the weights at their initial values to isolate the contribution of the
online adaptation.
