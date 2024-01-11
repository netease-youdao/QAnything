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
echo "Model directories check passed. (0/8)"
echo "模型路径检查通过. (0/8)"

# start llm server
nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble --http-port=10000 --grpc-port=10001 --metrics-port=10002 > /model_repos/QAEnsemble/QAEnsemble.log 2>&1 &

cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm transfer service is ready! (1/8)"
echo "大模型中转服务已就绪! (1/8)"

cd /workspace/qanything_local
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready! (2/8)"
echo "rerank服务已就绪! (2/8)"

nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready! (3/8)"
echo "OCR服务已就绪! (3/8)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything backend service is ready! (4/8)"
echo "qanything后端服务已就绪! (4/8)"


cd /workspace/qanything_local/front_end
# 安装依赖
echo "Waiting for [npm run install]（5/8)"
npm install
echo "[npm run install] Installed successfully（5/8)"

# 指定文件路径
package_json="./package.json"
version_txt="./version.txt"
# 从 package.json 中提取版本号
package_version=$(grep -E '"version"\s*:' "$package_json" | awk -F'"' '{print $4}')

# 读取 version.txt 中的版本号
if [ -e "$version_txt" ]; then
    version=$(cat "$version_txt")
    # 判断是否等于 package.json 中的版本号
    if [ "$version" != "$package_version" ]; then
        echo "Waiting for [npm run build](6/8)"
        npm run build
        echo "[npm run build] build successfully(6/8)"
    fi
else
    # 如果没有version文件，表示是第一次build
    echo "Waiting for [npm run build](6/8)"
    npm run build
    echo "[npm run build] build successfully(6/8)"
fi

# 启动前端页面服务
nohup npm run serve 1>http-server.log 2>&1 &

# 监听前端页面服务
tail -f http-server.log &
while ! grep -q "Local:" http-server.log; do
    echo "Waiting for the front-end service to start..."
    echo "等待启动前端服务"
    sleep 1
done

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
    sleep 5
  fi
done

current_time=$(date +%s)
elapsed=$((current_time - start_time))  # 计算经过的时间（秒）
echo "Time elapsed: ${elapsed} seconds."
echo "已耗时: ${elapsed} 秒."
echo "Please visit the front-end service at [http://localhost:5052/qanything/] to conduct Q&A. Please replace "localhost" with the actual IP address according to the actual situation."
echo "请在[http://localhost:5052/qanything/]下访问前端服务来进行问答，请根据实际情况将localhost替换为实际ip"

# 保持容器运行
while true; do
  sleep 2
done


