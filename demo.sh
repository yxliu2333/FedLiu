#!/bin/bash

# 启动第一个会话记录
script -q -c "
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate fedmd_env
    python FEMNIST_Balanced.py -conf conf/EMNIST_balance_conf.json --beta 1.0 --gamma 1.0 --tauDIST 8.0 --tauKL 8.0
" 1_ba_1.0+1.0+8.0+8.0

# 启动第二个会话记录
script -q -c "
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate fedmd_env
    python FEMNIST_Imbalanced.py -conf conf/EMNIST_imbalance_conf.json --beta 1.0 --gamma 1.0 --tauDIST 8.0 --tauKL 8.0
" 1_im_1.0+1.0+8.0+8.0
