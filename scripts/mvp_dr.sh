#!/bin/bash

MODE="mvp"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet-r
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"

MEM_SIZE=2000
ONLINE_ITER=3
MODEL_NAME="mvp" 
EVAL_PERIOD=1000
BATCHSIZE=64
LR=5e-3 
OPT_NAME="adam" 
SCHED_NAME="default"

REPLAY_BETA=0.5
REPLAY_TAU=500

NOTE=MVP_DR_${DATASET}_MEM_${MEM_SIZE}_BS_${BATCHSIZE}_ITER_${ONLINE_ITER}_RBETA_${REPLAY_BETA}_RTAU_${REPLAY_TAU}

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
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir ../LargeMonitor/data \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --use_mask --use_contrastiv --use_afs --use_gsf > results/logs/mvp_dr_$DATASET_$NOTE_$seed.log 2>&1 &

    CUDA_IDX=$((CUDA_IDX + 1))
done