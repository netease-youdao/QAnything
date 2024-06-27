#!/bin/bash




update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file="/workspace/qanything_local/.env"

  # 检查键是否存在于.env文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi
}



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


script_name=$(basename "$0")

usage() {
  echo "Usage: $script_name [-d device] [-i <device_id>] [-h]"
  echo "  -d <device>: Specify argument [cpu、cuda、npu]"
  echo "  -i <device_id>: Specify argument GPU device_id"
  echo "  -h: Display help usage message"
  exit 1
}

device="cpu"
device_id="0"

# 解析命令行参数
while getopts ":d:i:h" opt; do
  case $opt in
    d) device=$OPTARG ;;
    i) device_id=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

echo "device is set to [$device]"
echo "device_id is set to [$device_id]"



cd /workspace/qanything_local

# 设置默认值
default_gpu_id1=0
default_gpu_id2=0

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
echo "GPU ID: $gpu_id1, $gpu_id2, device_id: $device_id"




start_time=$(date +%s)  # 记录开始时间

# 如果是nvidia显卡，则使用tritonserver
if [ "$device"=="cuda"]; then
  CUDA_VISIBLE_DEVICES=$gpu_id1 nohup /opt/tritonserver/bin/tritonserver --model-store=/workspace/qanything_local/model --http-port=9000 --grpc-port=9001 --metrics-port=9002 --log-verbose=1 > /workspace/qanything_local/logs/debug_logs/embed_rerank_tritonserver.log 2>&1 &
  update_or_append_to_env "RERANK_PORT" "9001"
  update_or_append_to_env "EMBED_PORT" "9001"

  cd /workspace/qanything_local || exit
  nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > /workspace/qanything_local/logs/debug_logs/rerank_server.log 2>&1 &
  echo "The rerank service is ready! (1/4)"
  echo "rerank服务已就绪! (1/4)"
fi


nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > /workspace/qanything_local/logs/debug_logs/ocr_server.log 2>&1 &
echo "The ocr service is ready! (2/4)"
echo "OCR服务已就绪! (2/4)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api_search.py --mode "local" > /workspace/qanything_local/logs/debug_logs/sanic_api_search.log 2>&1 &

# 监听后端服务启动
backend_start_time=$(date +%s)

while ! grep -q "Starting worker" /workspace/qanything_local/logs/debug_logs/sanic_api_search.log; do
    echo "Waiting for the backend service to start..."
    echo "等待启动后端服务"
    sleep 1

    # 获取当前时间并计算经过的时间
    current_time=$(date +%s)
    elapsed_time=$((current_time - backend_start_time))

    # 检查是否超时
    if [ $elapsed_time -ge 120 ]; then
        echo "启动后端服务超时，请检查日志文件 /workspace/qanything_local/logs/debug_logs/sanic_api_search.log 获取更多信息。"
        exit 1
    fi
    sleep 5
done


echo "The qanything backend service is ready! (3/4)"
echo "qanything后端服务已就绪! (3/4)"



embed_rerank_log_file="/workspace/qanything_local/logs/debug_logs/embed_rerank_tritonserver.log"
tail -f $embed_rerank_log_file &  # 后台输出日志文件
tail_pid=$!  # 获取tail命令的进程ID


if [ "$device"=="cuda"]; then
  now_time=$(date +%s)
  while true; do
      current_time=$(date +%s)
      elapsed_time=$((current_time - now_time))
      embed_rerank_response=$(curl -s -w "%{http_code}" http://localhost:9000/v2/health/ready -o /dev/null)

      

      # 检查是否超时
      if [ $elapsed_time -ge 120 ]; then
          kill $tail_pid  # 关闭后台的tail命令
          echo "启动 embedding and rerank 服务超时，自动检查 $embed_rerank_log_file 中是否存在Error..."
          check_log_errors "$embed_rerank_log_file"
          exit 1
      fi

      if [ $embed_rerank_response -eq 200 ]; then
          kill $tail_pid  # 关闭后台的tail命令
          echo "The embedding and rerank service is ready!. (4/4)"
          echo "Embedding 和 Rerank 服务已准备就绪！(4/4)"
          break
      fi

      echo "The embedding and rerank service is starting up, it can be long... you have time to make a coffee :)"
      echo "Embedding and Rerank 服务正在启动，可能需要一段时间...你有时间去冲杯咖啡 :)"
      sleep 10
  done

  check_log_errors "/workspace/qanything_local/logs/debug_logs/rerank_server.log"
fi





echo "开始检查日志文件中的错误信息..."
# 调用函数并传入日志文件路径
check_log_errors "/workspace/qanything_local/logs/debug_logs/ocr_server.log"
check_log_errors "/workspace/qanything_local/logs/debug_logs/sanic_api_search.log"

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."
user_ip=$USER_IP
echo "请在[http://$user_ip:8777/api/docs]下访问前端服务获取当前后端API文档说明"

# 保持容器运行
while true; do
  sleep 2
done