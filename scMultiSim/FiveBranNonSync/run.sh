#!/bin/bash
# 获取当前激活的conda环境名称
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')

# 检查当前环境是否为SynOT
if [ "$CURRENT_ENV" != "SynOT" ]; then
    echo "当前环境不是SynOT，正在切换..."

    # 方法1：对于conda初始化的shell
    # 注意：直接在脚本中使用conda activate通常不起作用，因为conda是shell函数
    # 需要先source conda的初始化脚本
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate SynOT

    # 如果还是有问题，可以尝试方法2：使用conda run
    # exec conda run -n SynOT "$0" "$@"  # 重新执行当前脚本但在SynOT环境中
fi

# 这里是脚本的其余部分，将在SynOT环境中执行
echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
nohup python -u RunEB.py >nohup.out 2>&1 &
