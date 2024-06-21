#!/bin/bash
echo "Script started at $(date)."
chmod +x "$0"

# 调用 Python 脚本并捕获输出
IFS=',' # 设置字段分隔符为逗号
openai_api_base_with_key=$(python config.py)

# 使用 read 命令分割字符串
read -r openai_api_base openai_api_key openai_api_model_name openai_api_context_length workers milvus_port qanything_port use_cpu <<< "$openai_api_base_with_key"
echo "openai_api_base: " $openai_api_base
echo "openai_api_key: " $openai_api_key
echo "openai_api_model_name: " $openai_api_model_name
echo "openai_api_context_length: " $openai_api_context_length
echo "workers: " $workers
echo "milvus_port: " $milvus_port
echo "qanything_port: " $qanything_port
echo "use_cpu: " $use_cpu


# 检查 Conda 是否安装，如果安装就执行一下逻辑，使用原有的conda进行安装
if command -v conda >/dev/null 2>&1; then
    echo "Conda is installed."
    # 检查 Conda 是否为最新版本
    echo "Checking for Conda updates..."
    if conda update --no-deps --dry-run -n base -c defaults conda | grep -q "will be updated"; then
        echo "An update is available for Conda."
        echo "是否更新 Codna 为新版本 (y/n)"
        read -r user_response

        if [[ "$user_response" =~ ^[Yy] ]]; then
            echo "Updating Conda..."
            conda update -n base -c defaults conda -y
            echo "Conda has been updated."
        else
            echo "Skipping Conda update."
        fi
    else
        echo "Conda is already up to date."
    fi


    # 检查特定 Conda 环境是否存在
    ENV_NAME="qanything-python"
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists."
    else
        echo "Conda environment '$ENV_NAME' does not exist. Proceeding with installation."
        conda create -n "$ENV_NAME" python=3.10
        if [ $? -eq 0 ]; then
            echo "Conda environment '$ENV_NAME' created successfully."
        else
            echo "Failed to create Conda environment '$ENV_NAME'."
            echo "创建conda环境失败 '$ENV_NAME'."
            exit 1
        fi
    fi
    # 激活 Conda 环境
    echo "Activating conda environment '$ENV_NAME'..."
    ENV_NAME="qanything-python"
    CONDA_INSTALL_PATH="$(conda info --base)"
    echo $CONDA_INSTALL_PATH
    chmod +x $CONDA_INSTALL_PATH
    # 使用从 Conda 获取的路径来激活 Conda 环境
    source "$CONDA_INSTALL_PATH/bin/activate" "$ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment '$ENV_NAME'."
        echo "激活conda环境失败 '$ENV_NAME'."
        exit 1
    fi
    echo "Conda environment '$ENV_NAME' activated."

    # 使用 pip 从 requirements.txt 安装依赖
    set -x
    set -e
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies from requirements.txt."
        exit 1
    fi
    # 判断操作系统
    if [ "$(uname)" = "Linux" ]; then
       S="LinuxOrWSL"
    else
        S="M1mac"
    fi
    use_openai_api_option="true"
    set -x
    echo "启动命令是scripts/base_run.sh -s "$S" -w "$workers" -m "$milvus_port" -q "$qanything_port" -o -b "$openai_api_base" -k "$openai_api_key" "
    bash scripts/base_run.sh -s "$S" -w "$workers" -m "$milvus_port" -q "$qanything_port" -o -b "$openai_api_base" -k "$openai_api_key"
    set +x
    set +e
    if [ $? -ne 0 ]; then
        echo "Failed to run the script for OpenAI API."
        exit 1
    fi
    echo "Script for OpenAI API executed successfully."
    # 在脚本结束时记录时间
    echo "Script finished at $(date)."
else
    echo "Conda is not installed."
fi


# 以下代码为用户未安装conda时执行的逻辑，该命令会在项目目录下安装一个conda并使用，该conda仅会在该脚本执行时间使用，不会填加到环境变量
# 检查当前使用的 shell
SHELL_NAME=$(basename "$SHELL")

# 设置对应的配置文件路径
case "$SHELL_NAME" in
    bash)
        CONFIG_FILE="~/.bashrc"
        ;;
    zsh)
        CONFIG_FILE="~/.zshrc"
        ;;
    *)
        CONFIG_FILE="~/.profile"  # 对于其他 shell，使用 .profile
        ;;
esac

echo "Config file for current shell is: $CONFIG_FILE"

# 获取当前脚本运行的目录
CURRENT_DIR=$(pwd)

# 定义 Conda 安装路径为当前目录下的 anaconda3 文件夹
CONDA_INSTALL_PATH="$CURRENT_DIR/anaconda3/bin"

# 将 Conda 的 bin 目录添加到 PATH 环境变量中
export PATH="$CONDA_INSTALL_PATH:$PATH"
#bash -c "source $CONFIG_FILE"

# 定义 Anaconda 安装程序的 URL，根据不同操作系统设置
if [ "$(uname)" = "Linux" ]; then
    INSTALLER_URL="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
    INSTALL_PATH="$(pwd)/anaconda3"
else
    # 这里假设如果不是 Linux，则操作系统为 macOS
    INSTALLER_URL="https://repo.anaconda.com/archive/Anaconda3-2024.02-1-MacOSX-x86_64.sh"
    INSTALL_PATH="$(pwd)/anaconda3"
fi


# 赋予执行权限
chmod +x "anaconda_installer.sh"

bash -c "source $CONFIG_FILE"
# 检测是否已安装 Conda 并询问用户是否更新
if command -v conda &> /dev/null; then
    echo "Local Conda is already installed."
    # 检查 Conda 是否为最新版本
    echo "Checking for Conda updates..."
    if conda update --no-deps --dry-run -n base -c defaults conda | grep -q "will be updated"; then
        echo "An update is available for Conda."
        echo "是否更新 Codna 为新版本 (y/n)"
        read -r user_response

        if [[ "$user_response" =~ ^[Yy] ]]; then
            echo "Updating Conda..."
            conda update -n base -c defaults conda -y
            echo "Conda has been updated."
        else
            echo "Skipping Conda update."
        fi
    else
        echo "Conda is already up to date."
    fi
else
    echo "Conda is not installed. Proceeding with a fresh installation."
    # 下载 Anaconda 安装程序
    echo "下载Anaconda安装程序到 $OS_TYPE..."
    wget "$INSTALLER_URL" -O "anaconda_installer.sh"

    # 提示用户输入以确认安装
    read -p "Do you want to install Anaconda in $INSTALL_PATH? (y/n) " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "Installing Anaconda..."
            chmod +x "anaconda_installer.sh"
           # 运行安装脚本
            yes | ./anaconda_installer.sh -b -p "$INSTALL_PATH"
            ;;
       *)
            echo "Installation aborted."
            exit 1
            ;;
    esac

    # 检查安装是否成功
    if conda --version &> /dev/null; then
        echo "Anaconda installation successful."
    else
       echo "Anaconda installation failed."
        exit 1
    fi

    # 仅在脚本运行期间设置局部环境变量，不修改全局环境变量
    echo "仅在脚本运行期间为conda设置局部环境变量，不修改全局环境变量"
    echo "Setting up local environment for Anaconda access."
    export PATH="$INSTALL_PATH/bin:$PATH"

    # 可以使用 conda -V 来验证 Conda 是否可用
    echo "Verifying Conda installation with conda -V..."
    conda -V

    # 获取当前脚本运行的目录，即项目路径
    PROJECT_DIR=$(pwd)

    # 定义 Conda 安装路径
#    CONDA_INSTALL_PATH="$PROJECT_DIR/anaconda3"

    # 定义 Conda 环境目录，使用项目路径下的 .conda/envs
    CONDA_ENVS_DIR="$PROJECT_DIR/.conda/envs"

    # 确保 Conda 环境目录存在
    if [ ! -d "$CONDA_ENVS_DIR" ]; then
        echo "Conda envs directory does not exist. Creating now at $CONDA_ENVS_DIR."
        mkdir -p "$CONDA_ENVS_DIR"
       if [ $? -ne 0 ]; then
           echo "Failed to create Conda envs directory."
           exit 1
       fi
    fi

    # 确保当前用户有写权限
    if [ ! -w "$CONDA_ENVS_DIR" ]; then
        echo "No write permission for the Conda envs directory. Attempting to fix permissions."
        chmod u+w "$CONDA_ENVS_DIR"
        if [ $? -ne 0 ]; then
            echo "Failed to modify permissions for Conda envs directory."
           exit 1
        fi
    fi
# 脚本结束时，局部环境变量 PATH 的更改将不再影响后续命令
fi


# 使用项目路径下的 Conda 安装来检查或创建环境
ENV_NAME="qanything-python"

# 检查环境是否存在
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    "$CONDA_INSTALL_PATH/conda" create -n "$ENV_NAME" python=3.10
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment '$ENV_NAME'."
        exit 1
    fi
    echo "Conda environment '$ENV_NAME' created successfully."
fi

# 激活 Conda 环境
echo "Activating conda environment '$ENV_NAME'..."
source "$CONDA_INSTALL_PATH/activate" "$ENV_NAME"
#conda activate qanything-python
echo "Conda environment created successfully."

CONDA_INSTALL_PATH_2="$CURRENT_DIR/anaconda3"
echo "$CONDA_PREFIX"
echo "CONDA_INSTALL_PATH_2/envs/qanything-python"
# 检查激活是否成功
if [ "$CONDA_PREFIX" != "$CONDA_INSTALL_PATH_2/envs/qanything-python" ]; then
    echo "Failed to activate conda environment."
    exit 1
fi


# 安装 requirements.txt 中的依赖
echo "Installing dependencies from requirements.txt..."
set -e  # 使脚本在遇到错误时立即退出
set -x  # 打印出执行的每一条命令

conda install pip
pip install -r requirements.txt

conda env list

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Check the error messages above for details."
    exit 1
fi

set +x  # 关闭命令打印
set +e  # 关闭立即退出

echo "Dependencies installed successfully."


# 使用 Conda 环境运行脚本
echo "Executing the script for OpenAI API in the 'qanything-python' environment."
# 判断操作系统
if [ "$(uname)" = "Linux" ]; then
    S="LinuxOrWSL"
else
    S="M1mac"
fi
use_openai_api_option="true"
set -x
echo "启动命令是scripts/base_run.sh -s "$S" -w "$workers" -m "$milvus_port" -q "$qanything_port" -o -b "$openai_api_base" -k "$openai_api_key" "
bash scripts/base_run.sh -s "$S" -w "$workers" -m "$milvus_port" -q "$qanything_port" -o -b "$openai_api_base" -k "$openai_api_key"
set +x
if [ $? -ne 0 ]; then
    echo "Failed to run the script for OpenAI API."
    exit 1
fi
echo "Script for OpenAI API executed successfully."


# 在脚本结束时记录时间
echo "Script finished at $(date)."