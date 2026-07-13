#!/usr/bin/env bash
# Launch a big Hebbian/CMA-ES run on ALC (RTX 4090). Detached, survives SSH drops.
# Usage (on remote):  setsid bash scripts/hebbian/run_big_alc.sh <history_len> <seed> &
set -u
cd /home/diggelen/AerialMan

HISTORY_LEN="${1:-1}"
SEED="${2:-2}"
# big population: VRAM is dominated by fixed sim overhead (~3GB), so envs scale cheaply.
OPTIMIZER="${OPTIMIZER:-cmaes}"   # cmaes | revde
POPSIZE="${POPSIZE:-512}"
N_REPEATS="${N_REPEATS:-8}"
# RevDE evaluates 3*popsize offspring per generation; size envs accordingly for full parallelism.
if [ "${OPTIMIZER}" = "revde" ]; then
    NUM_ENVS="${NUM_ENVS:-$((3 * POPSIZE * N_REPEATS))}"
else
    NUM_ENVS="${NUM_ENVS:-$((POPSIZE * N_REPEATS))}"
fi
NUM_GEN="${NUM_GEN:-400}"
DE_F="${DE_F:-0.5}"
DE_CR="${DE_CR:-0.9}"
WEIGHT_INIT_SCALE="${WEIGHT_INIT_SCALE:-0.01}"
EVOLVE_INIT="${EVOLVE_INIT:-0}"             # 1 -> add evolved initial-weights genome block (+W)
KEEP_D="${KEEP_D:-0}"                       # 1 -> keep D term too (with EVOLVE_INIT -> ABCD+W)
SIGMA="${SIGMA:-0.3}"                       # PGPE/CMA initial exploration std
BOUND="${BOUND:-1.0}"                       # ABC(+D)(+initW) genome bound [-BOUND, BOUND]
PGPE_CENTER_LR="${PGPE_CENTER_LR:-0.15}"
PGPE_STDEV_LR="${PGPE_STDEV_LR:-0.1}"
PGPE_STDEV_MAX_CHANGE="${PGPE_STDEV_MAX_CHANGE:-0.2}"
MAX_HOURS="${MAX_HOURS:-}"                  # optional wall-clock cap (hours)
UPDATE_INTERVAL="${UPDATE_INTERVAL:-1}"     # plasticity timescale: avg activations over N steps per ΔW
HEBBIAN_LR="${HEBBIAN_LR:-0.1}"             # ABC(D) online learning rate
REWARD_CURRICULUM="${REWARD_CURRICULUM:-0}" # 1 -> widen goal_range mid-training (task scheduling)
EVO_ARGS=""
[ "${EVOLVE_INIT}" = "1" ] && EVO_ARGS="--evolve_init_weights"
[ "${KEEP_D}" = "1" ] && EVO_ARGS="${EVO_ARGS} --keep_d"
[ "${REWARD_CURRICULUM}" = "1" ] && EVO_ARGS="${EVO_ARGS} --reward_curriculum"
[ -n "${MAX_HOURS}" ] && EVO_ARGS="${EVO_ARGS} --max_hours ${MAX_HOURS}"
WANDB="${WANDB:-0}"                          # 1 to enable Weights & Biases logging
WANDB_PROJECT="${WANDB_PROJECT:-hebbian-flycrane}"
WANDB_MODE="${WANDB_MODE:-online}"           # online | offline | disabled
WANDB_ARGS=""
if [ "${WANDB}" = "1" ]; then
    WANDB_ARGS="--wandb --wandb_project ${WANDB_PROJECT} --wandb_mode ${WANDB_MODE}"
    [ -n "${WANDB_ENTITY:-}" ] && WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
fi
# distinct stdout log per config so parallel sweep runs don't collide
ew=""
[ "${EVOLVE_INIT}" = "1" ] && ew="_ew"
[ "${EVOLVE_INIT}" = "1" ] && [ "${KEEP_D}" = "1" ] && ew="_ewd"
if [ "${OPTIMIZER}" = "revde" ]; then
    LOG="run_revde_F${DE_F}_CR${DE_CR}_p${POPSIZE}_i${WEIGHT_INIT_SCALE}${ew}_h${HISTORY_LEN}_seed${SEED}.log"
elif [ "${OPTIMIZER}" = "pgpe" ]; then
    LOG="run_pgpe_p${POPSIZE}r${N_REPEATS}_clr${PGPE_CENTER_LR}_slr${PGPE_STDEV_LR}_b${BOUND}${ew}_ui${UPDATE_INTERVAL}_h${HISTORY_LEN}_seed${SEED}.log"
else
    LOG="run_${OPTIMIZER}_p${POPSIZE}_i${WEIGHT_INIT_SCALE}${ew}_h${HISTORY_LEN}_seed${SEED}.log"
fi

# only kill existing runs when explicitly asked (so parallel runs can coexist)
if [ "${KILL_EXISTING:-0}" = "1" ]; then
    pkill -f "scripts/hebbian/train.py" 2>/dev/null || true
    sleep 3
fi

exec .venv/bin/python scripts/hebbian/train.py \
    --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
    --headless \
    --optimizer="${OPTIMIZER}" --de_F="${DE_F}" --de_CR="${DE_CR}" \
    --sigma="${SIGMA}" --bound="${BOUND}" \
    --pgpe_center_lr="${PGPE_CENTER_LR}" --pgpe_stdev_lr="${PGPE_STDEV_LR}" \
    --pgpe_stdev_max_change="${PGPE_STDEV_MAX_CHANGE}" \
    --weight_init_scale="${WEIGHT_INIT_SCALE}" --hebbian_update_interval="${UPDATE_INTERVAL}" \
    --hebbian_lr="${HEBBIAN_LR}" ${EVO_ARGS} \
    ${WANDB_ARGS} \
    --num_envs="${NUM_ENVS}" --popsize="${POPSIZE}" --n_repeats="${N_REPEATS}" \
    --num_generations="${NUM_GEN}" --history_len="${HISTORY_LEN}" --seed="${SEED}" \
    > "${LOG}" 2>&1
