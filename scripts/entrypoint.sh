#!/bin/bash

check_log_errors() {
    local log_file=$1  # 将第一个参数赋值给变量log_file，表示日志文件的路径

    # 检查日志文件是否存在
    if [[ ! -f "$log_file" ]]; then
        echo "指定的日志文件不存在: $log_file"
        return 1
    fi

    # 使用grep命令检查"core dumped"或"Error"的存在
    # -C 5表示打印匹配行的前后各5行
    local pattern="core dumped|Error|error"
    if grep -E -C 5 "$pattern" "$log_file"; then
        echo "检测到错误信息，请查看上面的输出。"
        exit 1
    else
        echo "$log_file 中未检测到明确的错误信息。请手动排查 $log_file 以获取更多信息。"
    fi
}

start_time=$(date +%s)  # 记录开始时间

# 设置默认值
default_gpu_id1=0
default_gpu_id2=1

# 检查环境变量GPUID1是否存在，并读取其值或使用默认值
if [ -z "${GPUID1}" ]; then
    gpu_id1=$default_gpu_id1
else
    gpu_id1=${GPUID1}
fi

# 检查环境变量GPUID2是否存在，并读取其值或使用默认值
if [ -z "${GPUID2}" ]; then
    gpu_id2=$default_gpu_id2
else
    gpu_id2=${GPUID2}
fi
echo "GPU ID: $gpu_id1, $gpu_id2"

DIR="/workspace/QAnything/logs/debug_logs"

# 检查目录是否存在
if [ ! -d "$DIR" ]; then
  # 如果目录不存在，则创建目录
  mkdir -p "$DIR"
  echo "Directory $DIR created."
else
  echo "Directory $DIR already exists."
fi

# 创建软连接
if [ ! -L "/workspace/QAnything/qanything_kernel/dependent_server/embedding_server/embedding_model_configs_v0.0.1" ]; then  # 如果不存在软连接
  cd /workspace/QAnything/qanything_kernel/dependent_server/embedding_server && ln -s /root/bce-embedding-base_v1 embedding_model_configs_v0.0.1  # 创建软连接
fi
if [ ! -L "/workspace/QAnything/qanything_kernel/dependent_server/rerank_server/rerank_model_configs_v0.0.1" ]; then  # 如果不存在软连接
  cd /workspace/QAnything/qanything_kernel/dependent_server/rerank_server && ln -s /root/bce-reranker-base_v1 rerank_model_configs_v0.0.1  # 创建软连接
fi
if [ ! -L "/workspace/QAnything/qanything_kernel/dependent_server/ocr_server/ocr_models" ]; then  # 如果不存在软连接
  cd /workspace/QAnything/qanything_kernel/dependent_server/ocr_server && ln -s /root/ocr_models .  # 创建软连接
fi
if [ ! -L "/workspace/QAnything/qanything_kernel/utils/loader/pdf_to_markdown/checkpoints" ]; then  # 如果不存在软连接
  cd /workspace/QAnything/qanything_kernel/utils/loader/pdf_to_markdown && ln -s /root/pdf_models checkpoints  # 创建软连接
fi

cd /workspace/QAnything

CUDA_VISIBLE_DEVICES=$gpu_id1 nohup python3 -u qanything_kernel/dependent_server/rerank_server/rerank_server.py > /workspace/QAnything/logs/debug_logs/rerank_server.log 2>&1 &
CUDA_VISIBLE_DEVICES=$gpu_id2 nohup python3 -u qanything_kernel/dependent_server/embedding_server/embedding_server.py > /workspace/QAnything/logs/debug_logs/embedding_server.log 2>&1 &
nohup python3 -u qanything_kernel/dependent_server/ocr_server/ocr_server.py > /workspace/QAnything/logs/debug_logs/ocr_server.log 2>&1 &
nohup python3 -u qanything_kernel/dependent_server/insert_files_serve/insert_files_server.py --port 8110 --workers 4 > /workspace/QAnything/logs/debug_logs/insert_files_server.log 2>&1 &
nohup python3 -u qanything_kernel/qanything_server/sanic_api.py --port 8777 --workers 4 > /workspace/QAnything/logs/debug_logs/main_server.log 2>&1 &

# 监听后端服务启动
backend_start_time=$(date +%s)

while ! grep -q "Starting worker" /workspace/QAnything/logs/debug_logs/main_server.log; do
    echo "Waiting for the backend service to start..."
    echo "等待启动后端服务"
    sleep 1

    # 获取当前时间并计算经过的时间
    current_time=$(date +%s)
    elapsed_time=$((current_time - backend_start_time))

    # 检查是否超时
    if [ $elapsed_time -ge 180 ]; then
        echo "启动后端服务超时，自动检查日志文件 /workspace/QAnything/logs/debug_logs/main_server.log："
        check_log_errors /workspace/QAnything/logs/debug_logs/main_server.log
        exit 1
    fi
    sleep 5
done

echo "qanything后端服务已就绪!"


current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."
user_ip=$USER_IP
echo "请在[http://$user_ip:8777/qanything/]下访问前端服务来进行问答，如果前端报错，请在浏览器按F12以获取更多报错信息"

# Keep the container running
while true; do
    sleep 5
done
