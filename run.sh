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

script_name=$(basename "$0")

usage() {
  echo "Usage: $script_name [-c <llm_api>] [-i <device_id>] [-b <runtime_backend>] [-m <model_name>] [-t <conv_template>] [-p <tensor_parallel>] [-r <gpu_memory_utilization>] [-h]"
  echo "  -c : Options {local, cloud} to specify the llm API mode, default is 'local'. If set to '-c cloud', please mannually set the environments {OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_MODEL_NAME, OPENAI_API_CONTEXT_LENGTH} into .env fisrt in run.sh"
  echo "  -i <device_id>: Specify argument GPU device_id"
  echo "  -b <runtime_backend>: Specify argument LLM inference runtime backend, options={default, hf, vllm}"
  echo "  -m <model_name>: Specify argument the path to load LLM model using FastChat serve API, options={Qwen-7B-Chat, deepseek-llm-7b-chat, ...}"
  echo "  -t <conv_template>: Specify argument the conversation template according to the LLM model when using FastChat serve API, options={qwen-7b-chat, deepseek-chat, ...}"
  echo "  -p <tensor_parallel>: Use options {1, 2} to set tensor parallel parameters for vllm backend when using FastChat serve API, default tensor_parallel=1"
  echo "  -r <gpu_memory_utilization>: Specify argument gpu_memory_utilization (0,1] for vllm backend when using FastChat serve API, default gpu_memory_utilization=0.81"
  echo "  -h: Display help usage message. For more information, please refer to docs/QAnything_Startup_Usage_README.md"

  echo '
| Service Startup Command                                                                 | GPUs | LLM Runtime Backend      | LLM model                        |
| --------------------------------------------------------------------------------------- | -----|--------------------------| -------------------------------- |
| ```bash ./run.sh -c cloud -i 0 -b default```                                            | 1    | OpenAI API               | OpenAI API                       |
| ```bash ./run.sh -c local -i 0 -b default```                                            | 1    | FasterTransformer        | Qwen-7B-QAnything                |
| ```bash ./run.sh -c local -i 0 -b hf -m MiniChat-2-3B -t minichat```                    | 1    | Huggingface Transformers | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0 -b vllm -m MiniChat-2-3B -t minichat -p 1 -r 0.81```     | 1    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b default```                                          | 2    | FasterTransformer        | Qwen-7B-QAnything                |
| ```bash ./run.sh -c local -i 0,1 -b hf -m MiniChat-2-3B -t minichat```                  | 2    | Huggingface Transformers | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b vllm -m MiniChat-2-3B -t minichat -p 1 -r 0.81```   | 2    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b vllm -m MiniChat-2-3B -t minichat -p 2 -r 0.81```   | 2    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |

Note: You can choose the most suitable Service Startup Command based on your own device conditions.
(1) Local Embedding/Rerank will run on device gpu_id_1 when setting "-i 0,1", otherwise using gpu_id_0 as default.
(2) When setting "-c cloud" that will use local Embedding/Rerank and OpenAI LLM API, which only requires about 4GB VRAM (recommend for GPU device VRAM <= 8GB).
(3) When you use OpenAI LLM API, you will be required to enter {OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_MODEL_NAME, OPENAI_API_CONTEXT_LENGTH} immediately.
(4) "-b hf" is the most recommended way for running public LLM inference for its compatibility but with poor performance.
(5) When you choose a public Chat LLM for QAnything system, you should take care of a more suitable **PROMPT_TEMPLATE** (/path/to/QAnything/qanything_kernel/configs/model_config.py) setting considering different LLM models.
'
  exit 1
}

# 检查master分支是否有新代码
# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 定义醒目的提示信息
print_important_notice() {
    echo -e "${YELLOW}====================================================${NC}"
    echo -e "${YELLOW}******************** 重要提示 ********************${NC}"
    echo -e "${YELLOW}====================================================${NC}"
    echo
    echo -e "${RED}检测到master分支有新的代码更新，如需体验最新的功能，可以手动执行 git pull 来同步最新的代码。${NC}"
    echo
    sleep 5
}

# 获取最新的远程仓库信息
git fetch origin master

# 获取本地master分支的最新提交
LOCAL=$(git rev-parse master)
# 获取远程master分支的最新提交
REMOTE=$(git rev-parse origin/master)

if [ $LOCAL != $REMOTE ]; then
    # 本地分支与远程分支不一致，需要更新
    print_important_notice
else
    echo -e "${GREEN}当前master分支已是最新，无需更新。${NC}"
fi


llm_api="local"
device_id="0"
runtime_backend="default"
model_name=""
conv_template=""
tensor_parallel=1
gpu_memory_utilization=0.81

# 解析命令行参数
while getopts ":c:i:b:m:t:p:r:h" opt; do
  case $opt in
    c) llm_api=$OPTARG ;;
    i) device_id=$OPTARG ;;
    b) runtime_backend=$OPTARG ;;
    m) model_name=$OPTARG ;;
    t) conv_template=$OPTARG ;;
    p) tensor_parallel=$OPTARG ;;
    r) gpu_memory_utilization=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

# 获取大模型B数
if [ $llm_api = 'cloud' ]; then
    model_size='0B'
elif [ $runtime_backend = 'default' ]; then
    model_size='7B'
else
    read -p "请输入您使用的大模型B数(示例：1.8B/3B/7B): " model_size
    # 检查是否合法，必须输入数字+B的形式，可以是小数
    if ! [[ $model_size =~ ^[0-9]+(\.[0-9]+)?B$ ]]; then
        echo "Invalid model size. Please enter a number like '1.8B' or '3B' or '7B'."
        exit 1
    fi
fi
echo "model_size=$model_size"

update_or_append_to_env "MODEL_SIZE" "$model_size"

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

# 检查GPU ID是否合法
if ! [[ $gpu_id1 =~ ^[0-9]+$ ]] || ! [[ $gpu_id2 =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU IDs. Please enter IDs like '0' or '0,1'."
    exit 1
fi

update_or_append_to_env "GPUID1" "$gpu_id1"
update_or_append_to_env "GPUID2" "$gpu_id2"



if [ $llm_api = 'cloud' ]; then
  need_input_openai_info=1
  OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
  # 如果.env中已存在OPENAI_API_KEY的值（不为空），则询问用户是否使用上次默认值：$OPENAI_API_KEY，$OPENAI_API_BASE, $OPENAI_API_MODEL_NAME, $OPENAI_API_CONTEXT_LENGTH
  if [ -n "$OPENAI_API_KEY" ]; then
    read -p "Do you want to use the previous OPENAI_API_KEY: $OPENAI_API_KEY? (yes/no) 是否使用上次的OPENAI_API_KEY: $OPENAI_API_KEY？(yes/no) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
    if [ "$use_previous" = "yes" ]; then
      need_input_openai_info=0
    fi
  fi
  if [ $need_input_openai_info -eq 1 ]; then
    read -p "Please enter OPENAI_API_KEY: " OPENAI_API_KEY
    read -p "Please enter OPENAI_API_BASE (default: https://api.openai.com/v1):" OPENAI_API_BASE
    read -p "Please enter OPENAI_API_MODEL_NAME (default: gpt-3.5-turbo):" OPENAI_API_MODEL_NAME
    read -p "Please enter OPENAI_API_CONTEXT_LENGTH (default: 4096):" OPENAI_API_CONTEXT_LENGTH

    if [ -z "$OPENAI_API_KEY" ]; then  # 如果OPENAI_API_KEY为空，则退出
    echo "OPENAI_API_KEY is empty, please enter OPENAI_API_KEY."
    exit 1
    fi
    if [ -z "$OPENAI_API_BASE" ]; then  # 如果OPENAI_API_BASE为空，则设置默认值
      OPENAI_API_BASE="https://api.openai.com/v1"
    fi
    if [ -z "$OPENAI_API_MODEL_NAME" ]; then  # 如果OPENAI_API_MODEL_NAME为空，则设置默认值
      OPENAI_API_MODEL_NAME="gpt-3.5-turbo"
    fi
    if [ -z "$OPENAI_API_CONTEXT_LENGTH" ]; then  # 如果OPENAI_API_CONTEXT_LENGTH为空，则设置默认值
      OPENAI_API_CONTEXT_LENGTH=4096
    fi

    update_or_append_to_env "OPENAI_API_KEY" "$OPENAI_API_KEY"
    update_or_append_to_env "OPENAI_API_BASE" "$OPENAI_API_BASE"
    update_or_append_to_env "OPENAI_API_MODEL_NAME" "$OPENAI_API_MODEL_NAME"
    update_or_append_to_env "OPENAI_API_CONTEXT_LENGTH" "$OPENAI_API_CONTEXT_LENGTH"
  else
    OPENAI_API_BASE=$(grep OPENAI_API_BASE .env | cut -d '=' -f2)
    OPENAI_API_MODEL_NAME=$(grep OPENAI_API_MODEL_NAME .env | cut -d '=' -f2)
    OPENAI_API_CONTEXT_LENGTH=$(grep OPENAI_API_CONTEXT_LENGTH .env | cut -d '=' -f2)
    echo "使用上次的配置："
    echo "OPENAI_API_KEY: $OPENAI_API_KEY"
    echo "OPENAI_API_BASE: $OPENAI_API_BASE"
    echo "OPENAI_API_MODEL_NAME: $OPENAI_API_MODEL_NAME"
    echo "OPENAI_API_CONTEXT_LENGTH: $OPENAI_API_CONTEXT_LENGTH"
  fi
fi

echo "llm_api is set to [$llm_api]"
echo "device_id is set to [$device_id]"
echo "runtime_backend is set to [$runtime_backend]"
echo "model_name is set to [$model_name]"
echo "conv_template is set to [$conv_template]"
echo "tensor_parallel is set to [$tensor_parallel]"
echo "gpu_memory_utilization is set to [$gpu_memory_utilization]"

update_or_append_to_env "LLM_API" "$llm_api"
update_or_append_to_env "DEVICE_ID" "$device_id"
update_or_append_to_env "RUNTIME_BACKEND" "$runtime_backend"
update_or_append_to_env "MODEL_NAME" "$model_name"
update_or_append_to_env "CONV_TEMPLATE" "$conv_template"
update_or_append_to_env "TP" "$tensor_parallel"
update_or_append_to_env "GPU_MEM_UTILI" "$gpu_memory_utilization"

# 读取环境变量中的用户信息
source .env

# 检查是否存在USER_IP
if [ -z "${USER_IP}" ]; then
    # 如果USER_IP不存在，询问用户并保存配置
    read -p "Are you running the code on a remote server or on your local machine? (remotelocal) 您是在云服务器上还是本地机器上启动代码？(remote/local) " answer
    if [[ $answer == "local" || $answer == "本地" ]]; then
        ip="localhost"
    else
        read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " ip
        echo "当前设置的远程服务器IP地址为 $ip, QAnything启动后，本地前端服务（浏览器打开[http://$ip:5052/qanything/]）将远程访问[http://$ip:8777]上的后端服务，请知悉！"
        sleep 5
    fi

    # 保存配置    
    update_or_append_to_env "USER_IP" "$ip"

else
    # 读取上次的配置
    ip=$USER_IP
    read -p "Do you want to use the previous ip: $ip? (yes/no) 是否使用上次的ip: $host？(yes/no) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
    if [[ $use_previous != "yes" && $use_previous != "是" ]]; then
        read -p "Are you running the code on a remote server or on your local machine? (remote/local) 您是在远程服务器上还是本地机器上启动代码？(remote/local) " answer
        if [[ $answer == "local" || $answer == "本地" ]]; then
            ip="localhost"
        else
            read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " ip
            echo "当前设置的远程服务器IP地址为 $ip, QAnything启动后，本地前端服务（浏览器打开[http://$ip:5052/qanything/]）将远程访问[http://$ip:8777]上的后端服务，请知悉！"
            sleep 5
        fi
        # 保存新的配置
        update_or_append_to_env "USER_IP" "$ip"
    fi
fi

if [ -e /proc/version ]; then
  if grep -qi microsoft /proc/version || grep -qi MINGW /proc/version; then
    if grep -qi microsoft /proc/version; then
        echo "Running under WSL"
        if [ -z "${WIN_VERSION}" ]; then
            read -p "请输入Windows版本（WIN11/WIN10）回车默认选WIN11，请输入：" win_version
            win_version=${win_version:-WIN11}
            if [[ $win_version == "WIN11" || $win_version == "WIN10" ]]; then
                update_or_append_to_env "WIN_VERSION" "$win_version"
            else
                echo "目前只支持WIN11和WIN10，请选择其一输入"
                exit 1
            fi
        fi
        # win10系统不支持qanything-7b模型
        if [[ $WIN_VERSION == "WIN10" ]]; then
          if [[ $runtime_backend == "default" && $llm_api == "local" ]] || [[ $model_name == "Qwen-7B-QAnything" ]]; then
              echo "当前系统为Windows 10，不支持Qwen-7B-QAnything模型，请重新选择其他模型，可参考：docs/QAnything_Startup_Usage_README.md"
              exit 1
          fi
        fi
    else
        echo "Running under git bash"
    fi
    
    if docker-compose -p user -f docker-compose-windows.yaml down |& tee /dev/tty | grep -q "services.qanything_local.deploy.resources.reservations value 'devices' does not match any of the regexes"; then
        echo "检测到 Docker Compose 版本过低，请升级到v2.23.3或更高版本。执行docker-compose -v查看版本。"
    fi
    docker-compose -p user -f docker-compose-windows.yaml up -d
    docker-compose -p user -f docker-compose-windows.yaml logs -f qanything_local
  else
    echo "Running under native Linux"
    if docker-compose -p user -f docker-compose-linux.yaml down |& tee /dev/tty | grep -q "services.qanything_local.deploy.resources.reservations value 'devices' does not match any of the regexes"; then
        echo "检测到 Docker Compose 版本过低，请升级到v2.23.3或更高版本。执行docker-compose -v查看版本。"
    fi
    docker-compose -p user -f docker-compose-linux.yaml up -d
    docker-compose -p user -f docker-compose-linux.yaml logs -f qanything_local
    # 检查日志输出
  fi
else
  echo "/proc/version 文件不存在。请确认自己位于Linux或Windows的WSL环境下"
fi
