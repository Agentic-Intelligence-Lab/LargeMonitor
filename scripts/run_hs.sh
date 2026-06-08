CUDA_VISIBLE_DEVICES=0,1,2,3
python disjoint_dino_cka.py  \
  --data-dir ./data \
  --num-tasks 10 \
  --batch-size 64 \
  --model-tag vitb16 \
  --buffer-size 1024 \
  --output results/cka/imagenet_hs_dino_vitb16_cka_buf_1024_tasks_10_bs_64.csv