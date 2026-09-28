#!/usr/bin/env bash
# Quantitative head-to-head of every controller in the presentation, under ONE identical protocol
# (same seed, same goal sequence, same metric definitions). Headless, no rendering -> many envs, fast.
#
#   bash presentation/run_eval.sh
#
# Appends to presentation/results/comparison.csv:
#   model, checkpoint, obs_dim, train_best, eval_ret, eval_ret_std, pos_err_m, ori_err_rad, success, n, seed
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=./.venv/bin/python
CK=presentation/checkpoints
OUT=presentation/results/comparison.csv
LOG=presentation/logs
mkdir -p presentation/results "$LOG"

FIXED=Isaac-flycrane-payload-decentralized-hovering-v0
RAND=Isaac-flycrane-payload-decentralized-hovering-rand-v0
COMMON="--headless --num_envs 128 --episodes 6 --seed 123 --out $OUT"

ev () {  # tag  model  checkpoint  task  [extra...]
    local tag=$1 model=$2 ckpt=$3 task=$4; shift 4
    if [ ! -f "$ckpt" ]; then echo "[skip] $tag -- missing $ckpt"; return; fi
    echo "[eval] $tag ($(date +%H:%M:%S))"
    $PY scripts/hebbian/compare_paper_vs_hebbian.py $COMMON --model "$model" \
        --checkpoint "$ckpt" --task "$task" "$@" > "$LOG/eval_$tag.log" 2>&1
    echo "[eval] $tag rc=$? : $(grep -o '\[COMPARE\].*' "$LOG/eval_$tag.log" | tail -1)"
}

# fixed attachment
ev fixed_ABCD    hebbian "$CK/hebbian_ABCD_fixed_fit95.pt"   "$FIXED"
ev fixed_ABCW    hebbian "$CK/hebbian_ABCW_fixed_fit184.pt"  "$FIXED"
ev fixed_ABCDW   hebbian "$CK/hebbian_ABCDW_fixed_fit190.pt" "$FIXED"
ev fixed_mappo   paper   "$CK/mappo_paper_3_mav.pt"          "$FIXED"
# randomized attachment (per-env cable geometry)
ev rand_mappo    paper   "$CK/mappo_paper_3_mav.pt"          "$RAND" --attachment_per_env 1
ev rand_hebbian  hebbian "$CK/hebbian_ABCDW_rand_fit167.pt"  "$RAND" --attachment_per_env 1
# training fitness is a biased selector under per-env randomization, so every DR checkpoint is scored
ev rand_hebbian_ui100 hebbian "$CK/hebbian_ABCDW_rand_ui100_fit163.pt" "$RAND" --attachment_per_env 1
ev rand_hebbian_ABCW  hebbian "$CK/hebbian_ABCW_rand_ui100_fit158.pt"  "$RAND" --attachment_per_env 1
ev rand_hebbian_fixedtrained hebbian "$CK/hebbian_ABCDW_fixed_fit190.pt" "$RAND" --attachment_per_env 1

echo "[eval] ALL DONE $(date +%H:%M:%S)"
column -s, -t "$OUT"
