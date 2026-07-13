#!/usr/bin/env bash
# Wave 2 of the init-scale sweep: waits for wave 1 (init_scale=0) to finish, then launches the
# same 4 configs with init_scale=1. (8 Isaac Sim instances can't share 24GB, so we run 2 waves.)
cd /home/diggelen/AerialMan

echo "[wave2] waiting for wave 1 to finish..."
while [ "$(pgrep -cf '[s]cripts/hebbian/train.py')" -gt 0 ]; do
    sleep 60
done
echo "[wave2] wave 1 done; launching init_scale=1 runs"

for spec in "openai_es:256" "openai_es:128" "pgpe:512" "pgpe:128"; do
    opt="${spec%%:*}"; pop="${spec#*:}"
    OPTIMIZER="$opt" POPSIZE="$pop" N_REPEATS=8 WEIGHT_INIT_SCALE=1 \
        setsid bash scripts/hebbian/run_big_alc.sh 1 2 >/dev/null 2>&1 < /dev/null &
    echo "[wave2] launched $opt p$pop i1"
    sleep 45   # stagger Isaac Sim inits to avoid simultaneous memory spikes
done
echo "[wave2] all wave-2 runs launched"
