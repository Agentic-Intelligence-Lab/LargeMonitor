#!/bin/bash

MODE="ewc++"
DATASET="imagenet-hs" # cifar10, cifar100, tinyimagenet, imagenet-r, imagenet-hs
N_TASKS=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

MEM_SIZE=2000 ONLINE_ITER=3
REG_COEF=100
MODEL_NAME="vit_base" EVAL_PERIOD=1000
BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
NOTE=EWCpp_${DATASET}_T${N_TASKS}_MEM_${MEM_SIZE}_BS_${BATCHSIZE}_ITER_${ONLINE_ITER}_REG_${REG_COEF}

CUDA_IDX=0
for seed in 1 2 3 # 4 5
do
    export CUDA_VISIBLE_DEVICES=$CUDA_IDX
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE --reg_coef $REG_COEF \
    $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 > results/logs/${NOTE}_${seed}.log 2>&1 &

    CUDA_IDX=$((CUDA_IDX + 1))
done
