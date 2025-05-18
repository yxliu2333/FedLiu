#!/bin/bash

# 定义 beta 和 gamma 的组合
beta_gamma_combinations=(
  "1.0 1.0"
)

# 定义 tauDIST 和 tauKL 的组合
tau_combinations=(
  "0.5 0.5"
  "1.0 1.0"
  "2.0 2.0"
  "4.0 4.0"
  "8.0 8.0"
)

# 遍历所有组合
for beta_gamma in "${beta_gamma_combinations[@]}"; do
  read beta gamma <<< "$beta_gamma"
  for tau_pair in "${tau_combinations[@]}"; do
    read tauDIST tauKL <<< "$tau_pair"
    
    # 构造输出文件名
    output_file="1st_im_${beta}+${gamma}+${tauDIST}+${tauKL}"
    
    # 执行命令并记录输出
    script -q -c "
      # 初始化 Conda
      source ~/anaconda3/etc/profile.d/conda.sh
      conda activate fedmd_env
      python FEMNIST_Imbalanced.py -conf conf/EMNIST_imbalance_conf.json --beta $beta --gamma $gamma --tauDIST $tauDIST --tauKL $tauKL
    " "$output_file"
  done
done
