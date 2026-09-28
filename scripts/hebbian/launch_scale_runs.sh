#!/usr/bin/env bash
# Three follow-up evolution runs on the FIXED-attachment transport task, launched once the 4090 is free.
#
# Motivation: every previous run was stopped by its wall-clock cap while its best-so-far was still climbing
# (the 190.7 run hit max_hours=10 at generation 169 of a requested 400). The hook-grasp campaign hit the same
# wall and the fix there was simply more training, so these runs give the search a real budget.
#
# All settings are copied from the best fixed-attachment run to date
# (logs/hebbian/.../openai_es/2026-07-05_18-33-50_seed2, fitness 190.7); only the budget and the one
# ablated axis differ per arm.
#
#   A1 scale_ABCDW     - the 190.7 recipe with 900 generations / 24 h   -> does scale alone move it?
#   A2 matched_ABCD    - identical, minus the evolved initial weights   -> a FAIR ABCD baseline
#   A3 scale_ABCDW_H3  - identical, with a 3-frame observation history  -> the hook's frame-stack lever
#
# Lane A runs A1 then A3; lane B runs A2 concurrently (two concurrent runs are known to fit this 4090).
set -uo pipefail
cd "$HOME/AerialMan"
export PYTHONUNBUFFERED=1

PY=./.venv/bin/python
LOG="$HOME/AerialMan/experiments/scale_runs"
mkdir -p "$LOG"

# ---- wait for the GPU to free up -------------------------------------------------------------------
echo "[wait] $(date '+%F %T') waiting for the 4090 (need < 3000 MiB used and no training process)"
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    busy=$(pgrep -fc "python.*(train_backbone|experiment_policy|train\.py)" 2>/dev/null || echo 0)
    if [ "${used:-99999}" -lt 3000 ] && [ "${busy:-1}" -eq 0 ]; then
        echo "[wait] $(date '+%F %T') GPU free (${used} MiB) -- launching"
        break
    fi
    sleep 300
done

# ---- settings shared by all three arms (the 190.7 recipe) ------------------------------------------
COMMON="--task Isaac-flycrane-payload-decentralized-hovering-v0 --headless --device cuda:0 \
--optimizer openai_es --popsize 512 --n_repeats 8 --num_envs 4096 \
--latent_size 4 --hebbian_lr 0.05 --hebbian_update_interval 100 \
--bound 5.0 --sigma 0.3 --es_sigma 0.1 --es_lr 0.05 \
--obs_norm zscore --action_scale 2.0 --weight_init_scale 0.0 --seed 2 \
--num_generations 900 --max_hours 24 \
--log_root experiments --wandb --wandb_project hebbian-flycrane-scale"

run_arm () {  # exp_name  extra-args...
    local name=$1; shift
    echo "[run] $name START $(date '+%F %T')"
    $PY scripts/hebbian/train.py $COMMON --exp_name "$name" "$@" > "$LOG/$name.log" 2>&1
    echo "[run] $name END rc=$? $(date '+%F %T')"
}

lane_a () {
    # A1: the scale test -- the exact 190.7 recipe, just allowed to keep going
    run_arm scale_ABCDW    --evolve_init_weights --keep_d 1 --history_len 1
    # A3: the hook's frame-stack lever (obs 45 -> 135, genome 1000 -> 2800)
    run_arm scale_ABCDW_H3 --evolve_init_weights --keep_d 1 --history_len 3
    echo "[lane] A DONE $(date '+%F %T')"
}

lane_b () {
    sleep 180   # stagger the Isaac cold starts
    # A2: the matched ABCD baseline. Dropping --evolve_init_weights makes the D block mandatory
    # (hebbian_core: use_d = (not evolve_init_weights) or keep_d), so this is a 4-block ABCD genome
    # with the SAME optimizer, population, update interval and budget as A1 -- the comparison the
    # current ablation lacks.
    run_arm matched_ABCD   --keep_d 1 --history_len 1
    echo "[lane] B DONE $(date '+%F %T')"
}

lane_a &
lane_b &
wait
echo "[scale-runs] ALL DONE $(date '+%F %T')"
