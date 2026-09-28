#!/usr/bin/env bash
# Render the presentation clip set (sequentially -- one GPU). Re-running skips clips already rendered.
#
#   bash presentation/make_videos.sh            # default 600 control steps per clip
#   STEPS=400 bash presentation/make_videos.sh  # shorter clips
#
# The VK_ICD_FILENAMES / __NV_PRIME_* variables force Vulkan onto the discrete NVIDIA GPU; without them
# Isaac Sim's renderer hangs during init on this hybrid-GPU laptop. Every clip shares --seed 123, the same
# camera and the same rollout length, so they can be played side by side.
set -uo pipefail
cd "$(dirname "$0")/.."

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export PYTHONUNBUFFERED=1

PY=./.venv/bin/python
CK=presentation/checkpoints
OUT=presentation/videos
LOG=presentation/logs
COMMON="--headless --steps ${STEPS:-600} --frame_every 2 --fps 25 --seed 123"
mkdir -p "$OUT" "$LOG"

FIXED=Isaac-flycrane-payload-decentralized-hovering-v0
RAND=Isaac-flycrane-payload-decentralized-hovering-rand-v0

render () {  # name  policy  checkpoint  task  [extra args...]
    local name=$1 policy=$2 ckpt=$3 task=$4; shift 4
    if [ ! -f "$ckpt" ]; then
        echo "[skip] $name -- checkpoint missing: $ckpt"
        return
    fi
    if [ -s "$OUT/$name.mp4" ]; then
        echo "[keep] $name -- already rendered"
        return
    fi
    echo "[render] $name  ($(date +%H:%M:%S))"
    $PY presentation/render_video.py $COMMON --policy "$policy" --checkpoint "$ckpt" \
        --task "$task" "$@" --out "$OUT/$name.mp4" > "$LOG/$name.log" 2>&1
    echo "[render] $name done rc=$? ($(date +%H:%M:%S))"
}

# --- Part 1: FIXED attachment -- what the evolved genome has to encode ------------------------------
# NOTE: a pure ABC genome (3 blocks: no D term AND random initial weights) cannot be built by the
# current hebbian_core -- the D block is forced on whenever the initial weights are not evolved.
# See README.md ("What is missing").
render 1a_fixed_ABCD   hebbian "$CK/hebbian_ABCD_fixed_fit95.pt"   "$FIXED"
render 1b_fixed_ABCW   hebbian "$CK/hebbian_ABCW_fixed_fit184.pt"  "$FIXED"
render 1c_fixed_ABCDW  hebbian "$CK/hebbian_ABCDW_fixed_fit190.pt" "$FIXED"
# reference: the paper's MAPPO policy on the task it was trained for (fixed attachment, agent id valid)
render 1d_fixed_mappo  mappo   "$CK/mappo_paper_3_mav.pt"          "$FIXED"

# --- Part 2: RANDOMIZED attachment -- fixed agent identity vs online plasticity ---------------------
render 2a_rand_mappo   mappo   "$CK/mappo_paper_3_mav.pt"          "$RAND" --attachment_per_env 1
render 2b_rand_hebbian hebbian "$CK/hebbian_ABCDW_rand_fit167.pt"  "$RAND" --attachment_per_env 1
# reference: the fixed-attachment Hebbian genome dropped into the randomized task (no DR training)
render 2c_rand_hebbian_fixedtrained hebbian "$CK/hebbian_ABCDW_fixed_fit190.pt" "$RAND" --attachment_per_env 1

echo "[render] ALL DONE $(date +%H:%M:%S)"
ls -la "$OUT"
