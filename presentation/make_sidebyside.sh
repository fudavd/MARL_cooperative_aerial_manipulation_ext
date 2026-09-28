#!/usr/bin/env bash
# Build the side-by-side comparison clips for the slides (run after make_videos.sh).
#
#   bash presentation/make_sidebyside.sh
#
# Each output pairs two already-rendered clips horizontally with a caption burned in, so a single file can be
# dropped on a slide. Captions use ffmpeg's default font; drop the drawtext filter if it is unavailable.
set -uo pipefail
cd "$(dirname "$0")/.."
V=presentation/videos

pair () {  # out  left.mp4  left-caption  right.mp4  right-caption
    local out=$1 l=$2 lc=$3 r=$4 rc=$5
    if [ ! -s "$V/$l" ] || [ ! -s "$V/$r" ]; then
        echo "[skip] $out -- needs $l and $r"
        return
    fi
    echo "[pair] $out"
    ffmpeg -y -loglevel error \
        -i "$V/$l" -i "$V/$r" \
        -filter_complex "\
[0:v]scale=960:-2,drawtext=text='$lc':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[a];\
[1:v]scale=960:-2,drawtext=text='$rc':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[b];\
[a][b]hstack=inputs=2" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 "$V/$out"
}

# the headline comparison: randomized attachment, fixed agent identity vs online plasticity
pair sbs_rand_mappo_vs_hebbian.mp4 \
     2a_rand_mappo.mp4   "MAPPO (trained on fixed attachment)" \
     2b_rand_hebbian.mp4 "Hebbian ABCD+W (evolved plasticity)"

# the plasticity ablation: random initial weights vs evolved initial weights
pair sbs_fixed_ABCD_vs_ABCDW.mp4 \
     1a_fixed_ABCD.mp4  "ABCD - random initial weights (fitness 95)" \
     1c_fixed_ABCDW.mp4 "ABCD+W - evolved initial weights (fitness 191)"

# same controller, fixed task vs randomized task
pair sbs_mappo_fixed_vs_rand.mp4 \
     1d_fixed_mappo.mp4 "MAPPO - fixed attachment" \
     2a_rand_mappo.mp4  "MAPPO - randomized attachment"

ls -la "$V"/sbs_*.mp4 2>/dev/null
