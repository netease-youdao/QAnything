#!/bin/bash

# 函数：更新或追加键值对到.env文件
update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file=".env"

  # 如果不存在.env文件，则创建
  if [ ! -f "$env_file" ]; then
    touch "$env_file"
  fi

  # 确保文件以换行符结束
  sed -i'' -e '$a\' "$env_file"

  # 检查键是否存在于.env文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS (BSD sed)
      sed -i '' "/^${key}=/c\\
${key}=${value}" "$env_file"
    else
      # Linux (GNU sed)
      sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
    fi
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi

  # 再次确保文件以换行符结束
  sed -i'' -e '$a\' "$env_file"
}



# 检测支持的 Docker Compose 命令
if docker compose version &>/dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
elif docker-compose version &>/dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  echo "无法找到 'docker compose' 或 'docker-compose' 命令。"
  exit 1
fi

# 检查master分支是否有新代码
# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color


# 默认 device_id
device_id="-1"

usage() {
    echo "Usage: $0 [-i <device_id>]"
    echo " -i <device_id>: Specify GPU device_id"
    exit 1
}

while getopts "i:" opt; do
    case $opt in
        i) device_id=$OPTARG ;;
        *) usage ;;
    esac
done


# 检查device_id是否是0-9或-1
if [[ ! $device_id =~ ^[0-9]|-1$ ]]; then
    echo "device_id 必须是0-9或-1"
    exit 1
fi


echo "device_id=${device_id}"

# if device_id是-1，则提示在cpu上启动
if [[ $device_id == "-1" ]]; then
    echo "将在CPU上启动服务"
else
    echo "将在GPU $device_id 上启动服务"
fi

update_or_append_to_env "GPUID" "$device_id"

# 读取环境变量中的用户信息
source .env

# 检查是否存在USER_IP
if [ -z "${USER_IP}" ]; then
    # 如果USER_IP不存在，询问用户并保存配置
    read -p "Are you running the code on a remote server or on your local machine? (remote/local) 您是在云服务器上还是本地机器上启动代码？(remote/local) " answer
    if [[ $answer == "local" || $answer == "本地" ]]; then
        ip="localhost"
    else
        read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " ip
        echo "当前设置的远程服务器IP地址为 $ip, QAnything启动后，本地前端服务位于（浏览器打开[http://$ip:8777/qanything/]），请知悉！"
        sleep 5
    fi

    # 保存配置
    update_or_append_to_env "USER_IP" "$ip"

else
    # 读取上次的配置
    ip=$USER_IP
    read -p "Do you want to use the previous ip: $ip? (yes/no) 是否使用上次的ip: $ip ？(yes/no) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
    if [[ $use_previous != "yes" && $use_previous != "是" ]]; then
        read -p "Are you running the code on a remote server or on your local machine? (remote/local) 您是在远程服务器上还是本地机器上启动代码？(remote/local) " answer
        if [[ $answer == "local" || $answer == "本地" ]]; then
            ip="localhost"
        else
            read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " ip
            echo "当前设置的远程服务器IP地址为 $ip, QAnything启动后，本地前端服务位于（浏览器打开[http://$ip:8777/qanything/]），请知悉！"
            sleep 5
        fi
        # 保存新的配置
        update_or_append_to_env "USER_IP" "$ip"
    fi
fi

if [ -e /proc/version ]; then
  if grep -qi microsoft /proc/version || grep -qi MINGW /proc/version; then
    # 不支持Windows
    echo "当前版本不支持Windows，请在Linux环境下运行此脚本"
  else
    echo "Running under native Linux"
  if $DOCKER_COMPOSE_CMD -f docker-compose-linux.yaml down 2>&1 | tee /dev/tty | grep -q "services.qanything_local.deploy.resources.reservations value 'devices' does not match any of the regexes"; then
    echo "检测到 Docker Compose 版本过低，请升级到v2.23.3或更高版本。执行docker-compose -v查看版本。"
  fi

    # 如果不存在volumes，则创建
    if [ ! -d "volumes/es/data" ]; then
        mkdir -p volumes/es/data
        chmod 777 -R volumes/es/data
    fi

    $DOCKER_COMPOSE_CMD -f docker-compose-linux.yaml up -d
    $DOCKER_COMPOSE_CMD -f docker-compose-linux.yaml logs -f qanything_local
    # 检查日志输出
  fi
else
  echo "Running under Macos"
  if $DOCKER_COMPOSE_CMD -f docker-compose-mac.yaml down 2>&1 | tee /dev/tty | grep -q "services.qanything_local.deploy.resources.reservations value 'devices' does not match any of the regexes"; then
    echo "检测到 Docker Compose 版本过低，请升级到v2.23.3或更高版本。执行docker-compose -v查看版本。"
  fi

  $DOCKER_COMPOSE_CMD -f docker-compose-mac.yaml up -d
  $DOCKER_COMPOSE_CMD -f docker-compose-mac.yaml logs -f qanything_local
fi
