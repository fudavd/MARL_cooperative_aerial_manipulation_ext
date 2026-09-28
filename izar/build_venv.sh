#!/bin/bash
#SBATCH -A micro-515
#SBATCH --qos=build
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J venv_build
#SBATCH -o /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/venv_build.out
# Build the Isaac Sim 4.5 / IsaacLab 2.0.2 venv on izar (RHEL9, glibc 2.34, V100).
# Mirrors the local laptop venv: Python 3.10, torch 2.5.1+cu124, isaacsim 4.5.0, IsaacLab v2.0.2 source.
set -euo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
cd /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext
export PIP_NO_CACHE_DIR=1
export TMPDIR=/tmp/$USER; mkdir -p "$TMPDIR"
date
[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip wheel setuptools

echo "=== torch 2.5.1 + cu124 ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== Isaac Sim 4.5.0 ==="
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

echo "=== IsaacLab v2.0.2 source packages (editable) ==="
for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    pip install -e "thirdparty/IsaacLab/source/$p"
done

echo "=== project deps (pinned to laptop versions) ==="
pip install cma==4.4.4 rsl-rl-lib==2.3.3 wandb==0.28.0 tensorboard==2.20.0 tensorboardX==2.6.5 \
    moviepy==2.2.1 imageio-ffmpeg==0.6.0 opencv-python==4.11.0.86 scipy==1.15.3 matplotlib==3.10.9 \
    evotorch==0.6.1 h5py==3.16.0 hydra-core==1.3.3 omegaconf==2.3.1 gym==0.23.1 rl-games==1.6.1 stable_baselines3==2.8.0
echo "=== skrl fork (no deps) + project extension ==="
pip install -e thirdparty/skrl --no-deps
pip install -e exts/MARL_mav_carry_ext

echo "=== verify ==="
python - <<'PY'
import torch, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.version.cuda)
for p in ["isaacsim", "isaaclab", "skrl", "rsl-rl-lib", "cma", "numpy", "gymnasium", "MARL_mav_carry_ext"]:
    print(p, m.version(p))
PY
pip list 2>/dev/null | grep -iE "^(torch|isaacsim |isaaclab |skrl|numpy|gymnasium|rsl|cma|warp)" 
du -sh .venv
date; echo "=== BUILD DONE ==="
