#!/bin/bash

update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file="qanything_kernel/configs/model_config.py"

  # 检查键是否存在于配置文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi
}


# 初始化参数
system=""
milvus_port=19530
qanything_port=8777
model_size="7B"
use_cpu=false
use_openai_api=false
openai_api_base="https://api.openai.com/v1"
openai_api_key=""
openai_api_model_name="gpt-3.5-turbo-1106"
openai_api_context_length="4096"

# 使用getopts解析命令行参数
while getopts ":s:m:q:M:cob:k:n:l:w:" opt; do
  case $opt in
    s) system="$OPTARG"
    ;;
    m) milvus_port="$OPTARG"
    ;;
    q) qanything_port="$OPTARG"
    ;;
    M) model_size="$OPTARG"
    ;;
    c) use_cpu=true
    ;;
    o) use_openai_api=true
    ;;
    b) openai_api_base="$OPTARG"
    ;;
    k) openai_api_key="$OPTARG"
    ;;
    n) openai_api_model_name="$OPTARG"
    ;;
    l) openai_api_context_length="$OPTARG"
    ;;
    w) workers="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "$use_openai_api" = false ]; then
  openai_api_base=""
  openai_api_key=""
  openai_api_model_name=""
  openai_api_context_length=""
fi

# 确保必需参数已提供
if [ -z "$system" ] || [ -z "$milvus_port" ] || [ -z "$qanything_port" ]; then
    echo "必须提供 --system, --milvus_port 和 --qanything_port 参数。"
    exit 1
fi

# 如果$use_openai_api为true，那么检查是否提供了openai_api_key（是否不等于sk-xxx）
if [ "$use_openai_api" = true ] && [ "$openai_api_key" = "sk-xxx" ]; then
    echo "必须提供 --openai_api_key 参数，当前是：$openai_api_key。"
    exit 1
fi


if [ "$system" = "M1mac" ] && [ "$use_openai_api" = false ]; then
    # 检查 xcode-select 命令是否存在
    if ! command -v xcode-select &> /dev/null; then
        echo "xcode-select 命令不存在。请前往App Store下载Xcode。"
        # 结束脚本执行
        exit 1
    fi

    # 执行 xcode-select -p 获取当前Xcode路径
    xcode_path=$(xcode-select -p)

    # 检查 xcode-select 的输出是否以 /Applications 开头
    if [[ $xcode_path != /Applications* ]]; then
        echo "当前Xcode路径不是以 /Applications 开头。"
        echo "请确保你已从App Store下载了Xcode，如果已经下载，请执行以下命令："
        echo "sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
        exit 1
    else
        echo "Xcode 已正确安装在路径：$xcode_path"
    fi
fi

if [ "$use_cpu" = true ]; then
    use_cpu_option="--use_cpu"
else
    use_cpu_option=""
fi

if [ "$use_openai_api" = true ]; then
    use_openai_api_option="--use_openai_api"
else
    use_openai_api_option=""
fi

echo -e "即将启动后端服务，启动成功后请复制[\033[32mhttp://0.0.0.0:$qanything_port/qanything/\033[0m]到浏览器进行测试。"
echo "运行qanything-server的命令是："
echo "CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.qanything_server.sanic_api --host 0.0.0.0 --port $qanything_port --model_size $model_size $use_cpu_option $use_openai_api_option ${openai_api_base:+--openai_api_base "$openai_api_base"} ${openai_api_key:+--openai_api_key "$openai_api_key"} ${openai_api_model_name:+--openai_api_model_name "$openai_api_model_name"} ${openai_api_context_length:+--openai_api_context_length "$openai_api_context_length"} ${workers:+--workers "$workers"}"

sleep 5
# 启动qanything-server服务
CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.qanything_server.sanic_api --host 0.0.0.0 --port $qanything_port --model_size $model_size \
    $use_cpu_option $use_openai_api_option \
    ${openai_api_base:+--openai_api_base "$openai_api_base"} \
    ${openai_api_key:+--openai_api_key "$openai_api_key"} \
    ${openai_api_model_name:+--openai_api_model_name "$openai_api_model_name"} \
    ${openai_api_context_length:+--openai_api_context_length "$openai_api_context_length"} \
    ${workers:+--workers "$workers"}
