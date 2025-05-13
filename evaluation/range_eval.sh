#!/bin/bash

# 基础命令参数
BASE_CMD="torchrun --nproc_per_node=5 --nnodes=1 evaluation_script.py"
GPU="\"2,3,4,5,6\""
IPC="10"
CONFIG_PATH="../config/ipc10/voc2007.yaml"
BASE_LOAD_PATH="../results/condense/condense/voc2007/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20250421-0852/converted_data"

# 从1000到20000，步长为1000
for ((i=1000; i<=20000; i+=1000)); do
    # 构建完整的文件路径
    LOAD_FILE="converted_${i}.pt"
    FULL_LOAD_PATH="${BASE_LOAD_PATH}/${LOAD_FILE}"
    
    # 构建完整命令
    CMD="${BASE_CMD} --gpu=${GPU} --ipc=${IPC} --config_path=${CONFIG_PATH} --load_path=${FULL_LOAD_PATH}"
    
    # 打印并执行命令
    echo "Running: ${CMD}"
    eval ${CMD}
done