#!/bin/bash


### If using OpenAI API, please export the following environments into .env fisrt
echo "OPENAI_API_KEY=''" > .env
echo "OPENAI_API_BASE=''" >> .env
echo "OPENAI_API_MODEL_NAME='gpt-3.5-turbo'" >> .env
echo "OPENAI_API_CONTEXT_LENGTH=4096" >> .env

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

if [ $llm_api = 'cloud' ]; then
  echo "If set to '-c cloud', please mannually set the environments {OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_MODEL_NAME, OPENAI_API_CONTEXT_LENGTH} into .env fisrt in run.sh"
  read -p "Please enter OPENAI_API_KEY: " OPENAI_API_KEY
  read -p "Please enter OPENAI_API_BASE: " OPENAI_API_BASE
  read -p "Please enter OPENAI_API_MODEL_NAME: " OPENAI_API_MODEL_NAME
  read -p "Please enter OPENAI_API_CONTEXT_LENGTH: " OPENAI_API_CONTEXT_LENGTH
  
  echo "OPENAI_API_KEY='$OPENAI_API_KEY'" > .env
  echo "OPENAI_API_BASE='$OPENAI_API_BASE'" >> .env
  echo "OPENAI_API_MODEL_NAME='$OPENAI_API_MODEL_NAME'" >> .env
  echo "OPENAI_API_CONTEXT_LENGTH=$OPENAI_API_CONTEXT_LENGTH" >> .env
fi

echo "llm_api is set to [$llm_api]"
echo "device_id is set to [$device_id]"
echo "runtime_backend is set to [$runtime_backend]"
echo "model_name is set to [$model_name]"
echo "conv_template is set to [$conv_template]"
echo "tensor_parallel is set to [$tensor_parallel]"
echo "gpu_memory_utilization is set to [$gpu_memory_utilization]"

# 写入环境变量.env文件
echo "LLM_API=${llm_api}" >> .env
echo "DEVICE_ID=$device_id" >> .env
echo "RUNTIME_BACKEND=$runtime_backend" >> .env
echo "MODEL_NAME=$model_name" >> .env
echo "CONV_TEMPLATE=$conv_template" >> .env
echo "TP=$tensor_parallel" >> .env
echo "GPU_MEM_UTILI=$gpu_memory_utilization" >> .env

# 检查是否存在 models 文件夹
if [ ! -d "models" ]; then
  echo "models 文件夹不存在，开始克隆和解压模型..."
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
  git lfs install
  git clone https://www.wisemodel.cn/Netease_Youdao/qanything.git
  d_end_time=$(date +%s)
  elapsed=$((d_end_time - d_start_time))  # 计算经过的时间（秒）
  echo "Download Time elapsed: ${elapsed} seconds."
  echo "下载耗时: ${elapsed} 秒."

  # 解压模型文件
  unzip qanything/models.zip

  unzip_end_time=$(date +%s)
  elapsed=$((unzip_end_time - d_end_time))  # 计算经过的时间（秒）
  echo "unzip Time elapsed: ${elapsed} seconds."
  echo "解压耗时: ${elapsed} 秒."

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
if [[ -n "$device_id" ]]; then
    # 如果传入参数，分割成两个GPU ID
    IFS=',' read -ra gpu_ids <<< "$device_id"
    gpu_id1=${gpu_ids[0]}
    gpu_id2=${gpu_ids[1]:-$gpu_id1}  # 如果没有第二个ID，则默认使用第一个ID
fi

# 检查GPU ID是否合法
if ! [[ $gpu_id1 =~ ^[0-9]+$ ]] || ! [[ $gpu_id2 =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU IDs. Please enter IDs like '0' or '0,1'."
    exit 1
fi

echo "GPUID1=${gpu_id1}" >> .env
echo "GPUID2=${gpu_id2}" >> .env

# 检查是否存在用户文件
if [[ -f "$user_file" ]]; then
    # 读取上次的配置
    host=$(cat "$user_file")
    read -p "Do you want to use the previous host: $host? (yes/no) 是否使用上次的host: $host？(yes/no) 回车默认选yes，请输入:" use_previous
    use_previous=${use_previous:-yes}
    if [[ $use_previous != "yes" && $use_previous != "是" ]]; then
        read -p "Are you running the code on a cloud server or on your local machine? (cloud/local) 您是在云服务器上还是本地机器上启动代码？(cloud/local) " answer
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
    read -p "Are you running the code on a cloud server or on your local machine? (cloud/local) 您是在云服务器上还是本地机器上启动代码？(cloud/local) " answer
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
