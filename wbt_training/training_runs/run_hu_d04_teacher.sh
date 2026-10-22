#!/bin/bash
# Train HU_D04 terrain teacher with unit action scales.
# Required: REGISTRY=file:///absolute/bundle[,file:///another/bundle].
# Defaults: LOGGER=wandb, NGPUS=1; LOGGER=disabled keeps logging local.
# Optional: WANDB_PROJECT, WANDB_ENTITY, RUN_NAME.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
NGPUS="${NGPUS:-1}"
php_positive_integer NGPUS "$NGPUS"
php_prepare_inputs teacher ''
php_training_command
php_select_training_gpu
php_activate_sim
exec "${TRAIN_COMMAND[@]}" -m wbt_training.train_agent \
    exp:hu_d04_wbt_terrain_ref_noscale simulator:isaacsim \
    --training.registry-name "$REGISTRY" \
    "${LOGGER_ARGS[@]}" --logger.video.enabled False \
    "$@"
