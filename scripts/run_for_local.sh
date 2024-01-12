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

function build_front_end() {
    # 转到 front_end 目录
    cd /workspace/qanything_local/front_end || exit

    # 安装依赖
    echo "Waiting for [npm run install]（5/8)"
    npm install
    if [ $? -eq 0 ]; then
        echo "[npm run install] Installed successfully（5/8)"
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
}

# 在执行 build 前同时执行 git fetch，并提示用户是否需要 git pull
git fetch
# 检查远程是否有新的变动
remote_changes=$(git log HEAD..origin/master --oneline)
if [ -n "$remote_changes" ]; then
    # 如果有变动，提示用户
    echo "检测到远程仓库有新的变动。"
    read -t 10 -p "是否执行 git pull 更新本地仓库？\n[Y/N]: " user_input
    user_input=${user_input:-N} # 默认值为 'N'，如果用户没有输入任何内容

    # 根据用户的输入决定是否执行 git pull
    if [[ $user_input =~ ^[Yy]$ ]]; then
        echo "正在执行 git pull..."
        git pull
    else
        echo "跳过 git pull，继续执行脚本..."
    fi
else
    echo "远程仓库没有新的变动。"
fi

# 定义 commit.log 文件路径
commit_log_file="commit.log"

# 定义特定文件夹路径，这里假设是 front_end
folder_path="front_end/"

# 获取当前的 commit id
current_commit_id=$(git rev-parse HEAD)

# 判断 commit.log 文件是否存在
if [ ! -f "$commit_log_file" ]; then
    # 如果不存在，则执行 build 并创建 commit.log
    echo $current_commit_id > $commit_log_file
    build_front_end
else
    # 如果存在，则与当前的 commit id 比对
    saved_commit_id=$(cat $commit_log_file)
    if [ "$saved_commit_id" != "$current_commit_id" ]; then
        # 如果本地保存的 commit id 与当前的不一致，则执行 build 并更新 commit.log
        echo $current_commit_id > $commit_log_file
    	build_front_end
    else
        # 如果一致，则执行 git status 检查特定文件夹下是否有改动
        if git status --porcelain $folder_path | grep "^ M"; then
            # 如果存在 front_end 下的改动，则执行 build
    	    build_front_end
        else
            # 否则不执行 build
            echo "无需执行 build。"
        fi
    fi
fi

cd /workspace/qanything_local/front_end 
# 启动前端页面服务
nohup npm run serve 1>npm_server.log 2>&1 &

# 监听前端页面服务
tail -f npm_server.log &
while ! grep -q "Local:" npm_server.log; do
    echo "Waiting for the front-end service to start..."
    echo "等待启动前端服务"
    sleep 1
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


