#!/usr/bin/env bash
# Render the two SUCCESSFUL fixed-attachment Hebbian variants (ABC+W and ABCD+W) as longer, tighter-framed
# clips, then stack them side by side with captions for the slides.
#
#   bash presentation/make_fixed_success_video.sh
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
FIXED=Isaac-flycrane-payload-decentralized-hovering-v0
# longer rollout + closer camera than the survey clips in make_videos.sh.
# NB: never put a literal % in a drawtext caption -- ffmpeg eats it and the caption renders as nothing.
COMMON="--headless --steps ${STEPS:-1200} --frame_every 2 --fps 25 --seed 123 --cam_eye 3.0 3.0 2.0 --cam_lookat 0 0 0.9"
mkdir -p "$OUT" "$LOG"

one () {  # name  checkpoint
    echo "[render] $1 ($(date +%H:%M:%S))"
    $PY presentation/render_video.py $COMMON --policy hebbian --checkpoint "$2" \
        --task "$FIXED" --out "$OUT/$1.mp4" > "$LOG/$1.log" 2>&1
    echo "[render] $1 rc=$? ($(date +%H:%M:%S))"
}

one 3a_fixed_ABCW_success  "$CK/hebbian_ABCW_fixed_fit184.pt"
one 3b_fixed_ABCDW_success "$CK/hebbian_ABCDW_fixed_fit190.pt"

if [ -s "$OUT/3a_fixed_ABCW_success.mp4" ] && [ -s "$OUT/3b_fixed_ABCDW_success.mp4" ]; then
    echo "[pair] sbs_fixed_successful_hebbian.mp4"
    ffmpeg -y -loglevel error \
        -i "$OUT/3a_fixed_ABCW_success.mp4" -i "$OUT/3b_fixed_ABCDW_success.mp4" \
        -filter_complex "\
[0:v]scale=960:-2,drawtext=text='ABC+W  -  return 155  -  err 0.48 m  -  success 16 pct':x=24:y=22:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=12[a];\
[1:v]scale=960:-2,drawtext=text='ABCD+W  -  return 157  -  err 0.41 m  -  success 36 pct':x=24:y=22:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=12[b];\
[a][b]hstack=inputs=2" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/sbs_fixed_successful_hebbian.mp4"
fi
echo "[render] FIXED-SUCCESS SET DONE $(date +%H:%M:%S)"
ls -la "$OUT"/3?_fixed_*success*.mp4 "$OUT"/sbs_fixed_successful_hebbian.mp4 2>/dev/null
