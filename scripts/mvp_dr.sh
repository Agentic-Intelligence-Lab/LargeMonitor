#!/bin/bash

MODE="mvp"
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

ONLINE_ITER=3
MODEL_NAME="mvp" 
EVAL_PERIOD=1000
BATCHSIZE=64
LR=5e-3 
OPT_NAME="adam" 
SCHED_NAME="default"


DATASET="tinyimagenet" # cifar10, cifar100, tinyimagenet, imagenet-r
MEM_SIZE=0
REPLAY_BETA=1.0
REPLAY_TAU=200

NOTE=MVP_${DATASET}_MEM_${MEM_SIZE}_BS_${BATCHSIZE}_ITER_${ONLINE_ITER}_DR_RBETA_${REPLAY_BETA}_RTAU_${REPLAY_TAU}

CUDA_IDX=7
for seed in 1 2 3 #4 5
do
    export CUDA_VISIBLE_DEVICES=$CUDA_IDX

    python main.py --mode $MODE \
    --shift_replay_beta $REPLAY_BETA --shift_replay_tau $REPLAY_TAU \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ./data \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --use_mask --use_contrastiv --use_afs --use_gsf > results/logs/${NOTE}_${seed}.log 2>&1 &

    CUDA_IDX=$((CUDA_IDX + 1))
done