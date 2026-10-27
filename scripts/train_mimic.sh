#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
hps="mimic"
image_size=192
# Do not touch 
noise_schedule="linear"
sigma_small="False"
use_attn="False"
sampler_name="uniform"

dropout=0.0
cfg_train="True"
cfg_w=1.0

# Ablation study
cfg_dropout=0.1
cfg_dropout_style="cemb_only"  # Options: "cemb_only", "together", "independent"
global_cond="False"
use_semantic_encoder="False"
use_vae_encoder="False"
embedding_mode="concat_all"  # Options: "concat_cemb_semb", "semb_shift_scale", "add_all", "concat_all", "none"
time_embed_dim=512
cond_embed_dim=256
sem_embed_dim=0
kl_weight="1e-1"
kl_free_bits=1e-4
lambda_decor="0"

if [ "$use_attn" == "True" ]; then
    attention_resolutions="12"
else
    attention_resolutions="-1"
fi

exp_name="${hps}_${image_size}_model_${noise_schedule}_attn_${use_attn}_embedmode_${embedding_mode}_cfgdropstyle_${cfg_dropout_style}_glcon_${global_cond}_cfgdrp_${cfg_dropout}"

"${PYTHON:-python}" main.py \
    --exp_name=$exp_name \
    --hps $hps \
    --image_size $image_size \
    --in_channels 1 \
    --out_channels 1 \
    --model_channels 32 \
    --num_res_blocks 2 \
    --num_heads 4 \
    --num_heads_upsample -1 \
    --num_head_channels -1 \
    --conv_resample True \
    --channel_mult 1,2,3,4,5,6 \
    --dropout $dropout \
    --dims 2 \
    --use_checkpoint False \
    --attention_resolutions $attention_resolutions \
    --use_new_attention_order False \
    --embedding_mode $embedding_mode \
    --ema_rate 0.9999 \
    --steps 1000 \
    --sigma_small $sigma_small \
    --sampler_name $sampler_name \
    --noise_schedule $noise_schedule \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
    --timestep_respacing "ddim250" \
    --parents age race sex finding \
    --shuffle True \
    --num_workers 4 \
    --pin_memory True \
    --resume "" \
    --batch_size 64 \
    --lr 1e-4 \
    --wd 1e-4 \
    --adam_betas 0.9 0.9999 \
    --epochs 5000 \
    --eval_freq 1 \
    --start_epoch 0 \
    --accu_steps 1 \
    --grad_skip 10 \
    --grad_clip 1 \
    --plt_freq 5000 \
    --lr_warmup_steps 500 \
    --seed 7 \
    --deterministic False \
    --cfg_train $cfg_train \
    --cfg_dropout $cfg_dropout \
    --cfg_dropout_style $cfg_dropout_style \
    --cfg_w $cfg_w \
    --time_embed_dim $time_embed_dim \
    --cond_embed_dim $cond_embed_dim \
    --global_cond $global_cond \
    --parent_dims "{'sex': 1, 'age': 1, 'race': 3, 'finding': 1}" \
    --use_semantic_encoder $use_semantic_encoder \
    --sem_embed_dim $sem_embed_dim \
    --use_vae_encoder $use_vae_encoder \
    --enc_channel_mult 1,2,3,4,5,6,8 \
    --kl_weight $kl_weight \
    --kl_free_bits $kl_free_bits \
    --lambda_decor $lambda_decor \
    "$@"
