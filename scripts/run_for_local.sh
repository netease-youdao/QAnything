#!/bin/bash

env_file="/workspace/qanything_local/front_end/.env.production"
user_file="/workspace/qanything_local/user.config"

# 检查是否存在用户文件
if [[ -f "$user_file" ]]; then
    # 读取上次的配置
    host=$(cat "$user_file")
    read -p "Do you want to use the previous host $host? (yes/no) 是否使用上次的host $host？(是/否) " use_previous
    if [[ $use_previous != "yes" && $use_previous != "是" ]]; then
        read -p "Are you running the code on a cloud server or on your local machine? (cloud/local) 您是在云服务器上还是本地机器上启动代码？(云服务器/本地) " answer
        if [[ $answer == "local" || $answer == "本地" ]]; then
            host="localhost"
        else
            read -p "Please enter the server IP address 请输入服务器IP地址(示例：10.234.10.144): " host
        fi
        # 保存新的配置到用户文件
        echo "$host" > "$user_file"
    fi
else
    # 如果用户文件不存在，询问用户并保存配置
    read -p "Are you running the code on a cloud server or on your local machine? (cloud/local) 您是在云服务器上还是本地机器上启动代码？(云服务器/本地) " answer
    if [[ $answer == "local" || $answer == "本地" ]]; then
        host="localhost"
    else
        read -p "Please enter the server IP address 请输入服务器IP地址 [x.x.x.x]: " host
    fi
    # 保存配置到用户文件
    echo "$host" > "$user_file"
fi

# 保存IP地址到变量中
api_host="http://$host:8777"

# 使用 sed 命令更新 VITE_APP_API_HOST 的值
sed -i "s|VITE_APP_API_HOST=.*|VITE_APP_API_HOST=$api_host|" "$env_file"

echo "The file $env_file has been updated with the following configuration:"
grep "VITE_APP_API_HOST" "$env_file"

start_time=$(date +%s)  # 记录开始时间
# 检查模型文件夹是否存在
check_folder_existence() {
  if [ ! -d "/workspace/qanything_local/models/$1" ]; then
    echo "The $1 folder does not exist under /workspace/qanything_local/models/. Please check your setup."
    echo "在/workspace/qanything_local/models/下不存在$1文件夹。请检查您的设置。"
    exit 1
  fi
}

check_version_file() {
  local version_file="/workspace/qanything_local/models/version.txt"
  local expected_version="$1"

  # 检查 version.txt 文件是否存在
  if [ ! -f "$version_file" ]; then
    echo "/workspace/qanything_local/models/ 不存在version.txt 请检查您的模型文件。"
    exit 1
  fi

  # 读取 version.txt 文件中的版本号
  local version_in_file=$(cat "$version_file")

  # 检查版本号是否为 v2.1.0
  if [ "$version_in_file" != "$expected_version" ]; then
    echo "当前版本为 $version_in_file ，不是期望的 $expected_version 版本。请更新您的模型文件。"
    exit 1
  fi

  echo "检查模型版本成功，当前版本为 $expected_version。"
}

echo "Checking model directories..."
echo "检查模型目录..."
check_folder_existence "base"
check_folder_existence "embed"
check_folder_existence "rerank"
check_version_file "v2.1.0"
echo "Model directories check passed. (0/8)"
echo "模型路径和模型版本检查通过. (0/8)"

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

# start llm server
CUDA_VISIBLE_DEVICES=0 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_base --http-port=10000 --grpc-port=10001 --metrics-port=10002 --log-verbose=$VERBOSE > /model_repos/QAEnsemble_base/QAEnsemble_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_embed --http-port=9000 --grpc-port=9001 --metrics-port=9002 --log-verbose=$VERBOSE > /model_repos/QAEnsemble_embed/QAEnsemble_embed.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup /opt/tritonserver/bin/tritonserver --model-store=/model_repos/QAEnsemble_rerank --http-port=8000 --grpc-port=8001 --metrics-port=8002 --log-verbose=$VERBOSE > /model_repos/QAEnsemble_rerank/QAEnsemble_rerank.log 2>&1 &

cd /workspace/qanything_local/qanything_kernel/dependent_server/llm_for_local_serve || exit
nohup python3 -u llm_server_entrypoint.py --host="0.0.0.0" --port=36001 --model-path="tokenizer_assets" --model-url="0.0.0.0:10001" > llm.log 2>&1 &
echo "The llm transfer service is ready! (1/8)"
echo "大模型中转服务已就绪! (1/8)"

cd /workspace/qanything_local || exit
nohup python3 -u qanything_kernel/dependent_server/rerank_for_local_serve/rerank_server.py > rerank.log 2>&1 &
echo "The rerank service is ready! (2/8)"
echo "rerank服务已就绪! (2/8)"

CUDA_VISIBLE_DEVICES=0 nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ocr.log 2>&1 &
echo "The ocr service is ready! (3/8)"
echo "OCR服务已就绪! (3/8)"

nohup python3 -u qanything_kernel/qanything_server/sanic_api.py > api.log 2>&1 &
echo "The qanything backend service is ready! (4/8)"
echo "qanything后端服务已就绪! (4/8)"


# 转到 front_end 目录
cd /workspace/qanything_local/front_end || exit

# 安装依赖
echo "Waiting for [npm run install]（5/8)"
timeout_time=300
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
echo "Please visit the front-end service at [http://$host:5052/qanything/] to conduct Q&A."
echo "请在[http://$host:5052/qanything/]下访问前端服务来进行问答，如果前端报错，请在浏览器按F12以获取更多报错信息"

# 保持容器运行
while true; do
  sleep 2
done


