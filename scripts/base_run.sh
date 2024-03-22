#!/bin/bash

update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file="qanything_kernel/configs/model_config.py"

  # 检查键是否存在于配置文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi
}


# 初始化参数
system=""
milvus_port=19530
qanything_port=8777
model_size="7B"
use_cpu=false
use_openai_api=false
openai_api_base="https://api.openai.com/v1"
openai_api_key=""
openai_api_model_name="gpt-3.5-turbo-1106"
openai_api_context_length="4096"

# 使用getopts解析命令行参数
while getopts ":s:m:q:M:c:o:b:k:n:l:" opt; do
  case $opt in
    s) system="$OPTARG"
    ;;
    m) milvus_port="$OPTARG"
    ;;
    q) qanything_port="$OPTARG"
    ;;
    M) model_size="$OPTARG"
    ;;
    c) use_cpu=true
    ;;
    o) use_openai_api=true
    ;;
    b) openai_api_base="$OPTARG"
    ;;
    k) openai_api_key="$OPTARG"
    ;;
    n) openai_api_model_name="$OPTARG"
    ;;
    l) openai_api_context_length="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 确保必需参数已提供
if [ -z "$system" ] || [ -z "$milvus_port" ] || [ -z "$qanything_port" ]; then
    echo "必须提供 --system, --milvus_port 和 --qanything_port 参数。"
    exit 1
fi

# 如果$use_openai_api为true，那么检查是否提供了openai_api_key（是否不等于sk-xxx）
if [ "$use_openai_api" = true ] && [ "$openai_api_key" = "sk-xxx" ]; then
    echo "必须提供 --openai_api_key 参数，当前是：$openai_api_key。"
    exit 1
fi


need_start_milvus=false
if [ "$system" = "LinuxOrWSL" ]; then
    if ss -tuln | grep ":$milvus_port" > /dev/null; then
        echo "端口$milvus_port正在被监听。Milvus-Lite服务已启动。"
    else
        need_start_milvus=true
    fi
else
    if ss -tuln | grep ":$milvus_port" > /dev/null; then
        echo "端口$milvus_port正在被监听。Milvus-Lite服务已启动。"
    else
        need_start_milvus=true
    fi
fi

backend_start_time=$(date +%s)
if [ "$need_start_milvus" = true ]; then
    echo "端口$milvus_port没有被监听。"
    # 启动milvus服务
    nohup milvus-server --data milvus_data --proxy-port $milvus_port 1>milvus_server.log 2>&1 &
    tail -f milvus_server.log &
    while ! grep -q "Welcome to use Milvus!" milvus_server.log; do
        echo "Waiting for the Milvus-Lite service to start..."
        echo "等待启动Mivus-Lite服务"
        sleep 1

        # 获取当前时间并计算经过的时间
        current_time=$(date +%s)
        elapsed_time=$((current_time - backend_start_time))

        # 检查是否超时
        if [ $elapsed_time -ge 120 ]; then
            echo "启动Mivus-Lite服务超时，请检查日志文件milvus_server.log 获取更多信息。"
            exit 1
        fi
        sleep 5
    done
    echo "Milvus-Lite服务已启动。"
fi

if [ "$use_cpu" = true ]; then
    use_cpu_option="--use_cpu"
else
    use_cpu_option=""
fi

if [ "$use_openai_api" = true ]; then
    use_openai_api_option="--use_openai_api"
else
    use_openai_api_option=""
fi

# 启动qanything-server服务
backend_start_time=$(date +%s)
nohup qanything-server --host 0.0.0.0 --port $qanything_port --model_size $model_size \
    $use_cpu_option $use_openai_api_option \
    ${openai_api_base:+--openai_api_base "$openai_api_base"} \
    ${openai_api_key:+--openai_api_key "$openai_api_key"} \
    ${openai_api_model_name:+--openai_api_model_name "$openai_api_model_name"} \
    ${openai_api_context_length:+--openai_api_context_length "$openai_api_context_length"} \
    1>qanything_server.log 2>&1 &

sleep 5
tail -f logs/debug.log/debug.log &

while ! grep -q "Starting worker" qanything_server.log; do
    echo "Waiting for the backend service to start..."
    echo "等待启动后端服务"
    sleep 1

    # 获取当前时间并计算经过的时间
    current_time=$(date +%s)
    elapsed_time=$((current_time - backend_start_time))

    # 检查是否超时
    if [ $elapsed_time -ge 120 ]; then
        echo "启动后端服务超时，请检查日志文件 /workspace/qanything_local/logs/debug_logs/sanic_api.log 获取更多信息。"
        exit 1
    fi
    sleep 5
done
echo "后端服务已启动。"


