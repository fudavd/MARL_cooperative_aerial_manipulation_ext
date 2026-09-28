#!/usr/bin/env bash
# Score the configurations that the first sweep did not produce: the MAPPO fixed-attachment run (it hit a
# transient CUDA fault) and the alternative randomized-attachment Hebbian checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=./.venv/bin/python
CK=presentation/checkpoints
OUT=presentation/results/comparison.csv
LOG=presentation/logs
FIXED=Isaac-flycrane-payload-decentralized-hovering-v0
RAND=Isaac-flycrane-payload-decentralized-hovering-rand-v0
COMMON="--headless --num_envs ${ENVS:-96} --episodes 6 --seed 123 --out $OUT"

ev () {
    local tag=$1 model=$2 ckpt=$3 task=$4; shift 4
    [ -f "$ckpt" ] || { echo "[skip] $tag"; return; }
    echo "[eval] $tag ($(date +%H:%M:%S))"
    $PY scripts/hebbian/compare_paper_vs_hebbian.py $COMMON --model "$model" \
        --checkpoint "$ckpt" --task "$task" "$@" > "$LOG/eval_$tag.log" 2>&1
    echo "[eval] $tag rc=$? : $(grep -o '\[COMPARE\].*' "$LOG/eval_$tag.log" | tail -1)"
}

ev fixed_mappo        paper   "$CK/mappo_paper_3_mav.pt"               "$FIXED"
ev rand_hebbian_ui100 hebbian "$CK/hebbian_ABCDW_rand_ui100_fit163.pt" "$RAND" --attachment_per_env 1
echo "[eval] MISSING SET DONE $(date +%H:%M:%S)"

# Diagnostic: the MILDER randomization mode (per-run narrow arcs near the nominal corners) vs the
# per-env wide arcs used above. Distinguishes "the controller broke" from "this randomization is brutal".
ev rand_mild_hebbian hebbian "$CK/hebbian_ABCDW_rand_ui100_fit163.pt" "$RAND" --attachment_per_env 0
ev rand_mild_mappo   paper   "$CK/mappo_paper_3_mav.pt"               "$RAND" --attachment_per_env 0
echo "[eval] DIAGNOSTIC DONE $(date +%H:%M:%S)"
