#!/bin/bash

# 函数：更新或追加键值对到.env文件
update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file=".env"

  # 检查键是否存在于.env文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi
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

script_name=$(basename "$0")

usage() {
  echo "Usage: $script_name [-i <device_id>] [-h]"
  echo "  -d : device, '[cpu,cuda,npu]'"
  echo "  -i <device_id>: Specify argument GPU device_id"
  echo "  -h: Display help usage message. For more information, please refer to docs/QAnything_Startup_Usage_README.md"
  exit 1
}




device="cuda"
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



## 检查device_id是否合法,必须满足以下条件：
# 1. 以逗号分隔的多个数字
# 2. 每个数字必须是0或正整数
# 3. 不能有重复的数字
if ! [[ $device_id =~ ^([0-9]+,)*[0-9]+$ ]]; then
    echo "Invalid device_id. Please enter a comma-separated list of integers like '0' or '0,1'."
    exit 1
fi



gpu_id1=0
gpu_id2=0

# 判断命令行参数
if [[ -n "$device_id" ]]; then
    # 如果传入参数，分割成两个GPU ID
    IFS=',' read -ra gpu_ids <<< "$device_id"
    gpu_id1=${gpu_ids[0]}
    gpu_id2=${gpu_ids[1]:-$gpu_id1}  # 如果没有第二个ID，则默认使用第一个ID
fi

echo "GPUID1=${gpu_id1}, GPUID2=${gpu_id2}, device_id=${device_id}"
echo "device_id is set to [$device_id]"

update_or_append_to_env "GPUID1" "$gpu_id1"
update_or_append_to_env "GPUID2" "$gpu_id2"
update_or_append_to_env "DEVICE_ID" "$device_id"

# 读取环境变量中的用户信息
source .env


ip="localhost"
update_or_append_to_env "USER_IP" "$ip"


echo "Running under native Linux"
if $DOCKER_COMPOSE_CMD -p user -f docker-compose-linux-search.yaml down |& tee /dev/tty | grep -q "services.qanything_local.deploy.resources.reservations value 'devices' does not match any of the regexes"; then
    echo "检测到 Docker Compose 版本过低，请升级到v2.23.3或更高版本。执行docker-compose -v查看版本。"
fi
mkdir -p volumes/es/data
chmod 777 -R volumes/es/data
$DOCKER_COMPOSE_CMD -p user -f docker-compose-linux-search.yaml up -d
$DOCKER_COMPOSE_CMD -p user -f docker-compose-linux-search.yaml logs -f qanything_local
# 检查日志输出


