#!/bin/bash

MODE="mvp"
N_TASKS=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

MEM_SIZE=0
ONLINE_ITER=3
MODEL_NAME="mvp" 
EVAL_PERIOD=1000
BATCHSIZE=64
LR=5e-3 
OPT_NAME="adam" 
SCHED_NAME="default"

DATASET="imagenet-hs" # cifar10, cifar100, tinyimagenet, imagenet-r
# NOTE=MVP_${DATASET}_MEM_${MEM_SIZE}_BS_${BATCHSIZE}_ITER_${ONLINE_ITER}

CUDA_IDX=7

for MEM_SIZE in 0
do
    NOTE=MVP_${DATASET}_MEM_${MEM_SIZE}_BS_${BATCHSIZE}_ITER_${ONLINE_ITER}

    for seed in 1 2 3
    do
        export CUDA_VISIBLE_DEVICES=$CUDA_IDX

        python main.py --mode $MODE \
        --dataset $DATASET \
        --n_tasks $N_TASKS \
        --rnd_seed $seed \
        --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE \
        --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
        --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 \
        --use_mask --use_contrastiv --use_afs --use_gsf > results/logs/${NOTE}_${seed}.log 2>&1 &

        CUDA_IDX=$((CUDA_IDX + 1))
    done
done