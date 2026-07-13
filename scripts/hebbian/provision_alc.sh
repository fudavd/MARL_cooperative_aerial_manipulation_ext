#!/usr/bin/env bash
# Provision the Isaac Sim 4.5 / IsaacLab 2.0.2 environment on ALC (RTX 4090) for AerialMan.
# Mirrors the local venv: Python 3.10, torch 2.5.1+cu124, isaacsim 4.5.0, IsaacLab v2.0.2 source.
set -euo pipefail
cd /home/diggelen/AerialMan

source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^aerialman "; then
    conda create -y -n aerialman python=3.10
fi
conda activate aerialman
python -m pip install --upgrade pip

echo "=== torch 2.5.1 + cu124 (matches local) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== Isaac Sim 4.5.0 ==="
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com

echo "=== IsaacLab v2.0.2 source packages (editable) ==="
for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    pip install -e "thirdparty/IsaacLab/source/$p"
done

echo "=== cma + skrl fork + project extension ==="
pip install cma
pip install -e thirdparty/skrl
pip install -e exts/MARL_mav_carry_ext

echo "=== verify imports ==="
python - <<'PY'
import torch, isaacsim, isaaclab
print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import importlib.metadata as m
print("isaaclab", m.version("isaaclab"), "| isaacsim", m.version("isaacsim"))
import cma; print("cma", cma.__version__)
PY
echo "=== PROVISION DONE ==="
