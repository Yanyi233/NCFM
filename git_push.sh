#!/bin/bash

# 最大尝试次数（可根据需要调整）
MAX_ATTEMPTS=30
# 当前尝试次数
attempt=1

echo "开始尝试执行 git push main main..."

while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "尝试 #$attempt..."
    
    # 使用timeout命令限制执行时间为1秒
    if timeout 3 git push main main; then
        echo "git push 执行成功！"
        exit 0
    else
        # 检查是否是因为超时（退出码124）
        if [ $? -eq 124 ]; then
            echo "git push 超时（无响应）"
        else
            echo "git push 执行失败，可能是403错误或其他问题..."
        fi
        
        # 等待一段时间再重试（可根据需要调整等待时间）
        sleep 3
        ((attempt++))
    fi
done

echo "已达到最大尝试次数 ($MAX_ATTEMPTS)，放弃执行。"
exit 1