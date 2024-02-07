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
model_size_num=$(echo $model_size | grep -oP '^[0-9]+(\.[0-9]+)?')

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

# 获取显卡型号
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i $gpu_id1)
# nvidia RTX 30系列或40系列
gpu_series=$(echo $gpu_model | grep -oP 'RTX\s*(30|40)')
if ! command -v jq &> /dev/null; then
    echo "Error: jq 命令不存在，请使用 sudo apt update && sudo apt-get install jq 安装，再重新启动。"
    exit 1
fi
compute_capability=$(jq -r ".[\"$gpu_model\"]" scripts/gpu_capabilities.json)
# 如果compute_capability为空，则说明显卡型号不在gpu_capabilities.json中
if [ -z "$compute_capability" ]; then
    echo "您的显卡型号 $gpu_model 不在支持列表中，请联系技术支持。"
    exit 1
fi
echo "GPU1 Model: $gpu_model"
echo "Compute Capability: $compute_capability"

if ! command -v bc &> /dev/null; then
    echo "Error: bc 命令不存在，请使用 sudo apt update && sudo apt-get install bc 安装，再重新启动。"
    exit 1
fi

if [ $(echo "$compute_capability >= 7.5" | bc) -eq 1 ]; then
    OCR_USE_GPU="True"
else
    OCR_USE_GPU="False"
fi
echo "OCR_USE_GPU=$OCR_USE_GPU because $compute_capability >= 7.5"
update_or_append_to_env "OCR_USE_GPU" "$OCR_USE_GPU"

# 使用nvidia-smi命令获取GPU的显存大小（以MiB为单位）
GPU1_MEMORY_SIZE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id1)

OFFCUT_TOKEN=0
echo "===================================================="
echo "******************** 重要提示 ********************"
echo "===================================================="
echo ""

# 使用默认后端且model_size_num不为0
if [ "$runtime_backend" = "default" ] && [ "$model_size_num" -ne 0 ]; then
    if [ -z "$gpu_series" ]; then  # 不是Nvidia 30系列或40系列
        echo "默认后端为FasterTransformer，仅支持Nvidia RTX 30系列或40系列显卡，您的显卡型号为： $gpu_model, 不在支持列表中，将自动为您切换后端："
        # 如果显存大于等于24GB且计算力大于等于8.6，则可以使用vllm后端
        if [ "$GPU1_MEMORY_SIZE" -ge 24000 ] && [ $(echo "$compute_capability >= 8.6" | bc) -eq 1 ]; then
            echo "根据匹配算法，已自动为您切换为vllm后端（推荐）"
            runtime_backend="vllm"
        else
            # 自动切换huggingface后端
            echo "根据匹配算法，已自动为您切换为huggingface后端"
            runtime_backend="hf"
        fi
    fi
fi

if [ "$GPU1_MEMORY_SIZE" -lt 4000 ]; then # 显存小于4GB
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 不足以部署本项目，建议升级到GTX 1050Ti或以上级别的显卡"
    exit 1
elif [ "$model_size_num" -eq 0 ]; then  # 模型大小为0B, 表示使用openai api，4G显存就够了
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 可以使用在线的OpenAI API"
elif [ "$GPU1_MEMORY_SIZE" -lt 8000 ]; then  # 显存小于8GB
    # 显存小于8GB，仅推荐使用在线的OpenAI API
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 仅推荐使用在线的OpenAI API"
    if [ "$model_size_num" -gt 0 ]; then  # 模型大小大于0B
        echo "您的显存不足以部署 $model_size 模型，请重新选择模型大小"
        exit 1
    fi
elif [ "$GPU1_MEMORY_SIZE" -ge 8000 ] && [ "$GPU1_MEMORY_SIZE" -le 10000 ]; then  # 显存[8GB-10GB)
    # 8GB显存，推荐部署1.8B的大模型
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 推荐部署1.8B的大模型，包括在线的OpenAI API"
    if [ "$model_size_num" -gt 2 ]; then  # 模型大小大于2B
        echo "您的显存不足以部署 $model_size 模型，请重新选择模型大小"
        exit 1
    fi
elif [ "$GPU1_MEMORY_SIZE" -ge 10000 ] && [ "$GPU1_MEMORY_SIZE" -le 16000 ]; then  # 显存[10GB-16GB)
    # 10GB, 11GB, 12GB显存，推荐部署3B及3B以下的模型
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB，推荐部署3B及3B以下的模型，包括在线的OpenAI API"
    if [ "$model_size_num" -gt 3 ]; then  # 模型大小大于3B
        echo "您的显存不足以部署 $model_size 模型，请重新选择模型大小"
        exit 1
    fi
elif [ "$GPU1_MEMORY_SIZE" -ge 16000 ] && [ "$GPU1_MEMORY_SIZE" -le 22000 ]; then  # 显存[16-22GB)
    # 16GB显存
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 推荐部署小于等于7B的大模型"
    if [ "$model_size_num" -gt 7 ]; then  # 模型大小大于7B
        echo "您的显存不足以部署 $model_size 模型，请重新选择模型大小"
        exit 1
    fi
    if [ "$runtime_backend" = "default" ]; then  # 默认使用Qwen-7B-QAnything+FasterTransformer
        if [ -n "$gpu_series" ]; then
            # Nvidia 30系列或40系列
            if [ $gpu_id1 -eq $gpu_id2 ]; then
                echo "为了防止显存溢出，tokens上限默认设置为2700"
                OFFCUT_TOKEN=1400
            else
                echo "tokens上限默认设置为4096"
                OFFCUT_TOKEN=0
            fi
        else
            echo "您的显卡型号 $gpu_model 不支持部署Qwen-7B-QAnything模型"
            exit 1
        fi
    elif [ "$runtime_backend" = "hf" ]; then  # 使用Huggingface Transformers后端
        if [ "$model_size_num" -le 7 ] && [ "$model_size_num" -gt 3 ]; then  # 模型大小大于3B，小于等于7B
            if [ $gpu_id1 -eq $gpu_id2 ]; then
                echo "为了防止显存溢出，tokens上限默认设置为1400"
                OFFCUT_TOKEN=2700
            else
                echo "为了防止显存溢出，tokens上限默认设置为2300"
                OFFCUT_TOKEN=1800
            fi
        else
            echo "tokens上限默认设置为4096"
            OFFCUT_TOKEN=0
        fi
    elif [ "$runtime_backend" = "vllm" ]; then  # 使用VLLM后端
        if [ "$model_size_num" -gt 3 ]; then  # 模型大小大于3B
            echo "您的显存不足以使用vllm后端部署 $model_size 模型"
            exit 1
        else
            echo "tokens上限默认设置为4096"
            OFFCUT_TOKEN=0
        fi
    fi
elif [ "$GPU1_MEMORY_SIZE" -ge 22000 ] && [ "$GPU1_MEMORY_SIZE" -le 25000 ]; then  # [22GB, 24GB]
    echo "您当前的显存为 $GPU1_MEMORY_SIZE MiB 推荐部署7B模型"
    if [ "$model_size_num" -gt 7 ]; then  # 模型大小大于7B
        echo "您的显存不足以部署 $model_size 模型，请重新选择模型大小"
        exit 1
    fi
    OFFCUT_TOKEN=0
elif [ "$GPU1_MEMORY_SIZE" -gt 25000 ]; then  # 显存大于24GB
    OFFCUT_TOKEN=0
fi

update_or_append_to_env "OFFCUT_TOKEN" "$OFFCUT_TOKEN"

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

# 检查是否存在 models 文件夹，且models下是否存在embed，rerank，base三个文件夹
if [ ! -d "models" ] || [ ! -d "models/embed" ] || [ ! -d "models/rerank" ] || [ ! -d "models/base" ]; then
  echo "models文件夹不完整 开始克隆和解压模型..."
  echo "===================================================="
  echo "******************** 重要提示 ********************"
  echo "===================================================="
  echo ""
  echo "模型大小为8G左右，下载+解压时间可能较长，请耐心等待10分钟，"
  echo "仅首次启动需下载模型。"
  echo "The model size is about 8GB, the download and decompression time may be long, "
  echo "please wait patiently for 10 minutes."
  echo "Only the model needs to be downloaded for the first time."
  echo ""
  echo "===================================================="
  echo "如果你在下载过程中遇到任何问题，请及时联系技术支持。"
  echo "===================================================="
  # 记录下载和解压的时间

  d_start_time=$(date +%s)
  # 判断是否存在lfs，不存在建议使用sudo apt-get install git-lfs安装
  if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs 命令不存在，请使用 sudo apt update && sudo apt-get install git-lfs 安装。或参考 https://git-lfs.com/ 页面安装，再重新启动"
    exit 1
  fi

  # 如果存在QAanything/models.zip，不用下载
  if [ ! -f "QAnything/models.zip" ]; then
    echo "Downloading models.zip..."
    echo "开始下载模型文件..."
    git lfs install
    git clone https://www.modelscope.cn/netease-youdao/QAnything.git
    d_end_time=$(date +%s)
    elapsed=$((d_end_time - d_start_time))  # 计算经过的时间（秒）
    echo "Download Time elapsed: ${elapsed} seconds."
    echo "下载耗时: ${elapsed} 秒."
  else
    echo "models.zip already exists, no need to download."
    echo "models.zip已存在，无需下载。"
  fi

  # 解压模型文件
  # 判断是否存在unzip，不存在建议使用sudo apt-get install unzip安装
  if ! command -v unzip &> /dev/null; then
    echo "Error: unzip 命令不存在，请使用 sudo apt update && sudo apt-get install unzip 安装，再重新启动"
    exit 1
  fi

  unzip_start_time=$(date +%s)
  unzip QAnything/models.zip

  unzip_end_time=$(date +%s)
  elapsed=$((unzip_end_time - unzip_start_time))  # 计算经过的时间（秒）
  echo "unzip Time elapsed: ${elapsed} seconds."
  echo "解压耗时: ${elapsed} 秒."

  # 删除克隆的仓库
  # rm -rf QAnything
else
  echo "models 文件夹已存在，无需下载。"
fi

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

check_version_file "v2.1.0"
echo "Model directories check passed. (0/8)"
echo "模型路径和模型版本检查通过. (0/8)"

# 删除克隆的https://www.modelscope.cn/netease-youdao/QAnything.git模型仓库
rm -rf QAnything

user_file="user.config"

# 检查是否存在用户文件
if [[ -f "$user_file" ]]; then
    # 读取上次的配置
    host=$(cat "$user_file")
    read -p "Do you want to use the previous host: $host? (yes/no) 是否使用上次的host: $host？(yes/no) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
    if [[ $use_previous != "yes" && $use_previous != "是" ]]; then
        read -p "Are you running the code on a remote server or on your local machine? (remote/local) 您是在远程服务器上还是本地机器上启动代码？(remote/local) " answer
        if [[ $answer == "local" || $answer == "本地" ]]; then
            host="localhost"
        else
            read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " host
            echo "当前设置的远程服务器IP地址为 $host, QAnything启动后，本地前端服务（浏览器打开[http://$host:5052/qanything/]）将远程访问[http://$host:8777]上的后端服务，请知悉！"
            sleep 5
        fi
        # 保存新的配置到用户文件
        echo "$host" > "$user_file"
    fi
else
    # 如果用户文件不存在，询问用户并保存配置
    read -p "Are you running the code on a remote server or on your local machine? (remotelocal) 您是在云服务器上还是本地机器上启动代码？(remote/local) " answer
    if [[ $answer == "local" || $answer == "本地" ]]; then
        host="localhost"
    else
        read -p "Please enter the server IP address 请输入服务器公网IP地址(示例：10.234.10.144): " host
        echo "当前设置的远程服务器IP地址为 $host, QAnything启动后，本地前端服务（浏览器打开[http://$host:5052/qanything/]）将远程访问[http://$host:8777]上的后端服务，请知悉！"
        sleep 5
    fi
    # 保存配置到用户文件
    echo "$host" > "$user_file"
fi

if [ -e /proc/version ]; then
  if grep -qi microsoft /proc/version || grep -qi MINGW /proc/version; then
    if grep -qi microsoft /proc/version; then
        echo "Running under WSL"
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
