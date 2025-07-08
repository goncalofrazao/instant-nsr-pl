#!/bin/bash
#SBATCH --job-name=nerf
#SBATCH --output=neus_%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=rtx_3090:1
#SBATCH -c 16

CONFIG="configs/neus-blender.yaml"
DATASET_NAME="blendercues"
SCENE="lego2"
TAG="debug"
VAL_INTERVAL=500
MAX_STEPS=2000
NORMAL_LOSS=0.0
DEPTH_LOSS=0.0
NUM_VIEWS=100

python launch.py \
        --config "$CONFIG" \
        --gpu 0 \
        --train dataset.scene="$SCENE" \
        tag="$TAG" \
        trainer.val_check_interval=$VAL_INTERVAL \
        trainer.max_steps=$MAX_STEPS \
        dataset.name="$DATASET_NAME" \
        model.background_color=random \
        dataset.num_views=$NUM_VIEWS \
        system.loss.lambda_normal_l1=$NORMAL_LOSS \
        system.loss.lambda_normal_cos=$NORMAL_LOSS \
        system.loss.lambda_depth=$DEPTH_LOSS \
        system.loss.scale_depth=false \
        # system.loss.lambda_eikonal=0.0 \
        # system.loss.lambda_mask=1.0 \
        # system.loss.lambda_rgb_mse=0.0 \
        # model.geometry.mlp_network_config.n_neurons=256 \
        # model.geometry.mlp_network_config.n_hidden_layers=2 \
        # model.texture.mlp_network_config.n_neurons=128 \
        # model.texture.mlp_network_config.n_hidden_layers=2 \
        # system.optimizer.args.lr=0.0001 \
        
