#!/bin/bash

start_time=$(date +%s)  # 记录开始时间

mkdir -p /model_repos/QAEnsemble_base /model_repos/QAEnsemble_embed /model_repos/QAEnsemble_rerank /model_repos/monitor_logs
if [ ! -L "/model_repos/QAEnsemble_base/base" ]; then
  cd /model_repos/QAEnsemble_base && ln -s /model_repos/QAEnsemble/base .
fi

if [ ! -L "/model_repos/QAEnsemble_embed/embed" ]; then
  cd /model_repos/QAEnsemble_embed && ln -s /model_repos/QAEnsemble/embed .
fi

if [ ! -L "/model_repos/QAEnsemble_rerank/rerank" ]; then
  cd /model_repos/QAEnsemble_rerank && ln -s /model_repos/QAEnsemble/rerank .
fi

# 设置默认值
default_gpu_id1=0
default_gpu_id2=0

# 检查环境变量GPUID1是否存在，并读取其值或使用默认值
if [ -z "${GPUID1}" ]; then
    gpuid1=$default_gpu_id1
else
    gpuid1=${GPUID1}
fi

# 检查环境变量GPUID2是否存在，并读取其值或使用默认值
if [ -z "${GPUID2}" ]; then
    gpuid2=$default_gpu_id2
else
    gpuid2=${GPUID2}
fi
echo "GPU ID: $gpuid1, $gpuid2"

# start llm server
# 判断一下，如果gpu_id1和gpu_id2相同，则只启动一个triton_server
if [ $gpuid1 -eq $gpuid2 ]; then
    echo "The triton server will start on $gpuid1 GPU"
    CUDA_VISIBLE_DEVICES=$gpuid1 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble --http-port=10000 --grpc-port=10001 --metrics-port=10002 --log-verbose=1 > /model_repos/QAEnsemble/QAEnsemble.log 2>&1 &
    echo "RERANK_PORT=10001" >> /workspace/qanything_local/.env
    echo "EMBED_PORT=10001" >> /workspace/qanything_local/.env
else
    echo "The triton server will start on $gpuid1 and $gpuid2 GPUs"
    CUDA_VISIBLE_DEVICES=$gpuid1 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_base --http-port=10000 --grpc-port=10001 --metrics-port=10002 --log-verbose=1 > /model_repos/QAEnsemble_base/QAEnsemble_base.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpuid2 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_embed --http-port=9000 --grpc-port=9001 --metrics-port=9002 --log-verbose=1 > /model_repos/QAEnsemble_embed/QAEnsemble_embed.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$gpuid2 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_rerank --http-port=8000 --grpc-port=8001 --metrics-port=8002 --log-verbose=1 > /model_repos/QAEnsemble_rerank/QAEnsemble_rerank.log 2>&1 &
    echo "RERANK_PORT=8001" >> /workspace/qanything_local/.env
    echo "EMBED_PORT=9001" >> /workspace/qanything_local/.env
fi


# 默认ocr_use_gpu为True
OCR_USE_GPU="True"

# 使用nvidia-smi命令获取GPU的显存大小（以MiB为单位）
GPU1_MEMORY_SIZE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpuid1)
GPU2_MEMORY_SIZE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpuid2)

# 检查显存大小是否小于12G（即 12288 MiB）
if [ "$GPU2_MEMORY_SIZE" -lt 12288 ]; then
    OCR_USE_GPU="False"
fi

echo "GPU1_MEMORY_SIZE=$GPU1_MEMORY_SIZE"
echo "OCR_USE_GPU=$OCR_USE_GPU"

echo "===================================================="
echo "******************** 重要提示 ********************"
echo "===================================================="
echo ""

if [ "$GPU1_MEMORY_SIZE" -lt 8100 ]; then
    echo "检测到您的 GPU 显存小于 8GB，推荐使用 OpenAI 或其他在线大型语言模型 (LLM)。"
elif [ "$GPU1_MEMORY_SIZE" -ge 8100 ] && [ "$GPU1_MEMORY_SIZE" -le 16400 ]; then
    echo "检测到您的 GPU 显存在 8GB 到 16GB 之间，推荐使用本地 3B 大小以内的语言模型。"
else
    echo "检测到您的 GPU 显存大于 16GB，推荐使用本地 7B 的语言模型。"
fi

echo ""
echo "===================================================="
echo "请根据您的显存情况选择合适的语言模型以获得最佳性能。"
echo "===================================================="
echo ""
sleep 5


cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve || exit
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm transfer service is ready! (1/8)"
echo "大模型中转服务已就绪! (1/8)"

cd /workspace/qanything_local || exit
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready! (2/8)"
echo "rerank服务已就绪! (2/8)"

CUDA_VISIBLE_DEVICES=$gpuid2 nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py OCR_USE_GPU > ocr.log 2>&1 &
echo "The ocr service is ready! (3/8)"
echo "OCR服务已就绪! (3/8)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything backend service is ready! (4/8)"
echo "qanything后端服务已就绪! (4/8)"


timeout_time=300  # npm下载超时时间300秒，triton_server启动超时时间600秒

env_file="/workspace/qanything_local/front_end/.env.production"
user_file="/workspace/qanything_local/user.config"
user_ip=$(cat "$user_file")
# 读取env_file的第一行
current_host=$(grep VITE_APP_API_HOST "$env_file")
user_host="VITE_APP_API_HOST=http://$user_ip:8777"
# 检查current_host与user_host是否相同
if [ "$current_host" != "$user_host" ]; then
    # 使用 sed 命令更新 VITE_APP_API_HOST 的值
    sed -i "s|VITE_APP_API_HOST=.*|$user_host|" "$env_file"
    echo "The file $env_file has been updated with the following configuration:"
    grep "VITE_APP_API_HOST" "$env_file"
fi

# 转到 front_end 目录
cd /workspace/qanything_local/front_end || exit
# 安装依赖
echo "Waiting for [npm run install]（5/8)"
timeout $timeout_time npm install
if [ $? -eq 0 ]; then
    echo "[npm run install] Installed successfully（5/8)"
elif [ $? -eq 124 ]; then
    echo "npm install 下载超时，可能是网络问题，请修改 npm 代理。"
    exit 1
else
    echo "Failed to install npm dependencies."
    exit 1
fi

# 构建前端项目
echo "Waiting for [npm run build](6/8)"
npm run build
if [ $? -eq 0 ]; then
    echo "[npm run build] build successfully(6/8)"
else
    echo "Failed to build the front end."
    exit 1
fi

# 启动前端页面服务
nohup npm run serve 1>npm_server.log 2>&1 &

# 监听前端页面服务
tail -f npm_server.log &

front_end_start_time=$(date +%s)

while ! grep -q "Local:" npm_server.log; do
    echo "Waiting for the front-end service to start..."
    echo "等待启动前端服务"
    sleep 1

    # 获取当前时间并计算经过的时间
    current_time=$(date +%s)
    elapsed_time=$((current_time - front_end_start_time))

    # 检查是否超时
    if [ $elapsed_time -ge 120 ]; then
        echo "启动前端服务超时，请检查日志文件 front_end/npm_server.log 获取更多信息。"
        exit 1
    fi
done
echo "The front-end service is ready!...(7/8)"
echo "前端服务已就绪!...(7/8)"

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."

while true; do
  response=$(curl -s -w "%{http_code}" http://localhost:10000/v2/health/ready -o /dev/null)
  if [ $response -eq 200 ]; then
    echo "The triton service is ready!, now you can use the qanything service. (8/8)"
    echo "Triton服务已准备就绪！现在您可以使用qanything服务。（8/8)"
    break
  else
    echo "The triton service is starting up, it can be long... you have time to make a coffee :)"
    echo "Triton服务正在启动，可能需要一段时间...你有时间去冲杯咖啡 :)"

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # 检查是否超时
    if [ $elapsed_time -ge $((timeout_time * 2)) ]; then
        echo "启动Triton服务超时，请进入容器内检查/model_repos/QAEnsemble_base/QAEnsemble_base.log以获取更多信息。"
        exit 1
    fi
    sleep 5
  fi
done

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."
echo "Please visit the front-end service at [http://$user_ip:5052/qanything/] to conduct Q&A."
echo "请在[http://$user_ip:5052/qanything/]下访问前端服务来进行问答，如果前端报错，请在浏览器按F12以获取更多报错信息"

# 保持容器运行
while true; do
  sleep 2
done


