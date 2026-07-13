#!/usr/bin/env bash
# Wait for the current PGPE runs to finish, then launch the PGPE hyperparameter sweep.
# Run detached:  setsid bash scripts/hebbian/wait_and_sweep.sh > sweep_orchestrator.log 2>&1 &
cd /home/diggelen/AerialMan

echo "[wait_and_sweep] $(date): waiting for current train.py runs to finish..."
while [ "$(pgrep -cf '[s]cripts/hebbian/train.py')" -gt 0 ]; do
    sleep 120
done
echo "[wait_and_sweep] $(date): all runs finished. Launching sweep (4 agents x 6)."
bash scripts/hebbian/launch_sweep.sh 4 6
