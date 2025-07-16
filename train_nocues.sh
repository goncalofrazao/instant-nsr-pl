#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

CONFIG="configs/neus-blender.yaml"
DATASET_NAME="blendercues"
DATSET_CUES="gt"
SCENE="lego2"
VAL_INTERVAL=10000
MAX_STEPS=20000
NUM_VIEWS=100
TAG="nocues_${NUM_VIEWS}views"

python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train dataset.scene="$SCENE" \
        tag="$TAG" \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        dataset.name="$DATASET_NAME" \
        dataset.cues="$DATSET_CUES" \
        dataset.num_views=$NUM_VIEWS \
        model.radius=0.7 \
        system.loss.lambda_normal_l1=0.0 \
        system.loss.lambda_normal_cos=0.0 \
        system.loss.lambda_depth=0.0 \
        system.loss.scale_depth=true \
