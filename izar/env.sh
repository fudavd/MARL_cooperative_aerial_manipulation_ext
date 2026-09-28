# source this (login node or inside a job) to get the working Isaac Sim 4.5 / IsaacLab 2.0.2 environment on izar
module purge
module load gcc/11.3.0 python/3.10.4
export REPO=/home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext
source "$REPO/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES                 # Isaac Sim first-run EULA (headless, no prompt)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json   # never let Vulkan pick lvp/intel
export TMPDIR=/tmp/$USER; mkdir -p "$TMPDIR"
export PIP_NO_CACHE_DIR=1
export PATH=$HOME/bin:$PATH                     # git-lfs
cd "$REPO"
