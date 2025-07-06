#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

CONFIG="configs/neus-blender.yaml"
SCENE="lego2"
TAG="debug"
VAL_INTERVAL=500
MAX_STEPS=1000
NORMAL_LOSS=0.05
DEPTH_LOSS=0.1
NUM_VIEWS=6

python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train dataset.scene="$SCENE" \
        tag="$TAG" \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        dataset.name=blendercues \
        system.loss.lambda_normal_l1=$NORMAL_LOSS \
        system.loss.lambda_normal_cos=$NORMAL_LOSS \
        system.loss.lambda_depth=$DEPTH_LOSS \
        system.loss.scale_depth=true \
        model.background_color=random \
        dataset.num_views=$NUM_VIEWS

