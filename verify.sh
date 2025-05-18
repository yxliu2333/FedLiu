#!/bin/bash

for i in {1..10}; do
  # 启动第 i 轮的 Balanced 实验
  script -q -c "
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate fedmd_env
    python FEMNIST_Balanced.py \
      -conf conf/EMNIST_balance_conf.json \
      --beta 1.0 --gamma 1.0 --tauDIST 8.0 --tauKL 8.0
  " "${i}_ba_1.0+1.0+8.0+8.0"

  # 启动第 i 轮的 Imbalanced 实验
  script -q -c "
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate fedmd_env
    python FEMNIST_Imbalanced.py \
      -conf conf/EMNIST_imbalance_conf.json \
      --beta 1.0 --gamma 1.0 --tauDIST 8.0 --tauKL 8.0
  " "${i}_im_1.0+1.0+8.0+8.0"
done
