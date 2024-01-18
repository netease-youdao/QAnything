#!/bin/bash

# 检查是否存在 models 文件夹
if [ ! -d "models" ]; then
  echo "models 文件夹不存在，开始克隆和解压模型..."
  echo "模型大小为8G左右，下载+解压时间可能较长，请耐心等待15分钟，仅首次启动需下载模型"

  git lfs install
  git clone https://www.modelscope.cn/netease-youdao/QAnything.git
  # 解压模型文件
  unzip qanything/models.zip

  # 重命名解压后的模型文件夹
  if [ -d "model" ]; then
    mv "model" "models"
  elif [ -d "qanything/model" ]; then
    mv "qanything/model" "models"
  fi

  # 删除克隆的仓库
  rm -rf qanything
else
  echo "models 文件夹已存在，无需下载。"
fi

# 检查模型文件夹是否存在
check_folder_existence() {
  if [ ! -d "models/$1" ]; then
    echo "The $1 folder does not exist under QAnything/models/. Please check your setup."
    echo "在QAnything/models/下不存在$1文件夹。请检查您的模型文件。"
    exit 1
  fi
}

check_version_file() {
  local version_file="models/version.txt"
  local expected_version="$1"

  # 检查 version.txt 文件是否存在
  if [ ! -f "$version_file" ]; then
    echo "QAnything/models/ 不存在version.txt 请检查您的模型文件是否完整。"
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


#env_file="front_end/.env.production"
user_file="user.config"

gpu_id1=0
gpu_id2=0

# 判断命令行参数
if [[ -n "$1" ]]; then
    # 如果传入参数，分割成两个GPU ID
    IFS=',' read -ra gpu_ids <<< "$1"
    gpu_id1=${gpu_ids[0]}
    gpu_id2=${gpu_ids[1]:-$gpu_id1}  # 如果没有第二个ID，则默认使用第一个ID
fi

# 检查GPU ID是否合法
if ! [[ $gpu_id1 =~ ^[0-9]+$ ]] || ! [[ $gpu_id2 =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU IDs. Please enter IDs like '0' or '0,1'."
    exit 1
fi

# 检查是否存在用户文件
if [[ -f "$user_file" ]]; then
    # 读取上次的配置
    host=$(cat "$user_file")
    read -p "Do you want to use the previous host: $host? (yes/no) 是否使用上次的host: $host？(是/否) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
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
        read -p "Please enter the server IP address 请输入服务器IP地址(示例：10.234.10.144): " host
    fi
    # 保存配置到用户文件
    echo "$host" > "$user_file"
fi

# 保存IP地址到变量中
# api_host="http://$host:8777"

# 使用 sed 命令更新 VITE_APP_API_HOST 的值
# sed -i "s|VITE_APP_API_HOST=.*|VITE_APP_API_HOST=$api_host|" "$env_file"

# echo "The file $env_file has been updated with the following configuration:"
# grep "VITE_APP_API_HOST" "$env_file"

# 创建.env文件并写入环境变量
echo "GPUID1=${gpu_id1}" > .env
echo "GPUID2=${gpu_id2}" >> .env

docker-compose -p user -f docker-compose-linux.yaml down
docker-compose -p user -f docker-compose-linux.yaml up -d
docker-compose -p user -f docker-compose-linux.yaml logs -f qanything_local 
