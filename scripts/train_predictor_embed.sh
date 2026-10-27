#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


hps="embed" 

pgm_setup="sup_aux"
# pgm_setup="sup_pgm"
# pgm_setup="sup_determ"

image_size=192
use_manual="True"

if [ "$use_manual" == "True" ]; then
    exp_name="${hps}_${image_size}_${pgm_setup}"
else
    exp_name="${hps}_${image_size}_full_${pgm_setup}"
fi

"${PYTHON:-python}" -m causal_model.pgm.train_pgm \
    --exp_name=$exp_name \
    --hps $hps \
    --image_size=$image_size \
    --batch_size 64 \
    --parents density circle \
    --shuffle True \
    --num_workers 12 \
    --pin_memory True \
    --lr=1e-3 \
    --wd=0.05 \
    --epochs=5000 \
    --loss_norm l1 \
    --eval_freq 1 \
    --in_channels 1 \
    --pgm_setup=$pgm_setup \
    --enc_net CNN \
    --std_fixed 0 \
    --seed 7 \
    --deterministic False \
    --use_manual $use_manual \
    "$@"
