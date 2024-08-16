#!/bin/bash

# 定义要查找的进程名称
processes=("rerank_server.py" "embedding_server.py" "ocr_server.py" "insert_files_server.py" "sanic_api.py", "pdf_parser_server.py")

echo "进程名称 内存占用(MB)"
echo "--------------------"

for proc in "${processes[@]}"
do
    # 使用 pgrep 找到主进程的 PID
    pid=$(pgrep -f "$proc")
    
    if [ ! -z "$pid" ]
    then
        # 使用 ps 命令找到主进程及其所有子进程，并计算总内存使用
        mem=$(ps --ppid $pid -p $pid --no-headers -o rss | awk '{sum+=$1} END {print sum/1024}')
        
        printf "%-20s %0.2f\n" "$proc" "$mem"
    fi
done

