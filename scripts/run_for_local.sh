#!/bin/bash

start_time=$(date +%s)  # 记录开始时间
# 检查模型文件夹是否存在
check_folder_existence() {
  if [ ! -d "/workspace/qanything_local/models/$1" ]; then
    echo "The $1 folder does not exist under /workspace/qanything_local/models/. Please check your setup."
    echo "在/workspace/qanything_local/models/下不存在$1文件夹。请检查您的设置。"
    exit 1
  fi
}

echo "Checking model directories..."
echo "检查模型目录..."
check_folder_existence "base"
check_folder_existence "embed"
check_folder_existence "rerank"
echo "Model directories check passed. (0/7)"
echo "模型路径检查通过. (0/7)"

# start llm server
nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble --http-port=10000 --grpc-port=10001 --metrics-port=10002 > /model_repos/QAEnsemble/QAEnsemble.log 2>&1 &

cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm transfer service is ready! (1/7)"
echo "大模型中转服务已就绪! (1/7)"

cd /workspace/qanything_local
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready! (2/7)"
echo "rerank服务已就绪! (2/7)"

nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready! (3/7)"
echo "OCR服务已就绪! (3/7)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything backend service is ready! (4/7)"
echo "qanything后端服务已就绪! (4/7)"

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."

cd /workspace/qanything_local/front_end
# 安装依赖
nohup npm i -g yarn > npm_install_yarn.log 2>&1
nohup yarn > npm_install.log 2>&1
if [ $? -eq 0 ]; then
    echo "npm install completed, starting front_end development service... (5/7)"
    echo "npm install 完成，正在启动前端服务... (5/7)"
    # 安装完成后，启动前端服务
    nohup yarn dev > npm_run_dev.log 2>&1 &
    DEV_SERVER_PID=$!
    # echo "前端服务进程ID: $DEV_SERVER_PID"
    while ! grep -q "ready" npm_run_dev.log; do
        echo "Waiting for the front-end service to start..."
        echo "等待前端服务启动..."
        sleep 5
    done

    echo "The front-end service is ready!...(6/7)"
    echo "前端服务已就绪!...(6/7)"
else
    echo "npm install failed, please check the npm_install.log log file."
    echo "npm install 失败，请检查 npm_install.log 日志文件。"
    exit 1
fi

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."

while true; do
  response=$(curl -s -w "%{http_code}" http://localhost:10000/v2/health/ready -o /dev/null)
  if [ $response -eq 200 ]; then
    echo "The triton service is ready!, now you can use the qanything service. (7/7)"
    echo "Triton服务已准备就绪！现在您可以使用qanything服务。（7/7）"
    break
  else
    echo "The triton service is starting up, it can be long... you have time to make a coffee :)"
    echo "Triton服务正在启动，可能需要一段时间...你有时间去冲杯咖啡 :)"
    sleep 5
  fi
done

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."

# 保持容器运行
while true; do
  sleep 2
done
