docker版本安装攻略（Linux或Win11 WSL环境）

## 一、安装NVIDIA Driver （>=525.105.17）

### 步骤 1：准备工作

要检查已安装的 NVIDIA 驱动版本和 CUDA 版本，然后根据需要卸载旧版本，你可以遵循以下步骤：

### 检查 NVIDIA 驱动

1. **检查 NVIDIA 驱动版本**：
   打开终端，然后输入以下命令来检查当前安装的 NVIDIA 驱动版本：
   
   ```bash
   nvidia-smi
   ```
   
   这个命令将输出一些关于 NVIDIA GPU 的信息，包括安装的驱动版本。查看输出中的 "Driver Version"
   
   如果已经满足版本要求则可以跳过

### 如果需要卸载旧版本

如果检查结果显示 NVIDIA 驱动版本低于 525，你需要先卸载它们。

#### 卸载 NVIDIA 驱动

对于 NVIDIA 驱动，可以通过运行下列命令来卸载：

```bash
sudo apt-get purge nvidia-*
```

然后，重启你的计算机：

```bash
sudo reboot
```

### 更新依赖

1. **更新系统**：首先，更新你的系统包列表和系统本身，确保所有的依赖都是最新的。
   
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **安装必要的依赖**：
   
   ```bash
   sudo apt install build-essential
   ```

### 步骤 2：禁用 Nouveau 驱动（如果已卸载过旧版驱动可跳过）

Ubuntu 默认使用 Nouveau 驱动程序来支持 NVIDIA 的显卡。在安装官方 NVIDIA 驱动前，你需要禁用 Nouveau 驱动。

1. 打开 `/etc/modprobe.d/blacklist-nouveau.conf` 文件，并添加以下内容：
   
   ```bash
   sudo nano /etc/modprobe.d/blacklist-nouveau.conf
   ```
   
   然后添加：
   
   ```
   blacklist nouveau
   options nouveau modeset=0
   ```

2. 更新初始化 RAM 文件系统：
   
   ```bash
   sudo update-initramfs -u
   ```

3. 重启你的系统：
   
   ```bash
   sudo reboot
   ```

### 步骤 3：安装 NVIDIA 驱动

1. **添加 NVIDIA 的 PPA**（可选，但推荐）：
   
   ```bash
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt update
   ```

2. **安装 NVIDIA 驱动**：
   
   ```bash
   ubuntu-drivers devices  # 可用来查询可用的显卡驱动版本，选择大于等于525的版本
   sudo apt install nvidia-driver-535
   ```
   
   请将 `nvidia-driver-535` 替换为适合你的显卡的驱动版本。

3. **重启**：
   安装完成后，重启你的系统：
   
   ```bash
   sudo reboot
   ```

## 二、安装docker（>=20.10.5）和docker-compose（>=2.23.3）

### 步骤 1：检查已安装的 Docker 和 Docker Compose 版本

1. **检查 Docker 版本**：
   打开终端并输入以下命令：
   
   ```bash
   docker --version
   ```
   
   如果已安装 Docker，此命令将输出当前安装的 Docker 版本。

2. **检查 Docker Compose 版本**：
   对于 Docker Compose，运行：
   
   ```bash
   docker compose version
   ```
   
   或者，如果是使用旧版本的 Docker Compose（1.x 版本），可能需要运行：
   
   ```bash
   docker-compose -v
   ```

### 步骤 2：如果需要，卸载旧版本

如果已安装 Docker 或 Docker Compose 但版本不符合要求，则需要卸载它们。

1. **卸载 Docker**：
   
   ```bash
   sudo apt-get remove docker docker-engine docker.io containerd runc
   ```

2. **卸载 Docker Compose**：
   如果是通过包管理器安装的 Docker Compose，可以用 `apt-get` 移除：
   
   ```bash
   sudo apt-get remove docker-compose
   ```
   
   如果是手动安装（如放置在 `/usr/local/bin/docker-compose`），则需要手动删除：
   
   ```bash
   sudo rm /usr/local/bin/docker-compose
   ```

### 步骤 3：安装 Docker

确保安装版本大于 20.10.5。

1. **设置 Docker 仓库**：
   首先，更新 `apt` 包索引，并安装一些必需的包，这些包允许 `apt` 通过 HTTPS 使用仓库：
   
   ```bash
   sudo apt-get update
   sudo apt-get install \
     ca-certificates \
     curl \
     gnupg \
     lsb-release
   ```

2. **添加 Docker 的官方 GPG 密钥**：
   
   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   ```

3. **设置稳定版仓库**：
   
   ```bash
   echo \
   "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

4. **安装 Docker Engine**：
   
   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

5. **验证 Docker 安装**：
   
   ```bash
   sudo docker run hello-world
   ```
   
   这应该会下载一个测试镜像并在容器中运行它，证明 Docker 已正确安装。

### 步骤 4：安装 Docker Compose

确保安装版本大于 2.23.3。

1. **下载 Docker Compose**：
   你可以从 Docker 的 GitHub 仓库直接下载 Docker Compose 二进制文件。以安装版本 2.24.1 为例（请根据需要调整版本号）：
   
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   # 如果发现速度过慢，可以尝试使用下面的国内源
   sudo curl -SL https://mirror.ghproxy.com/https://github.com/docker/compose/releases/download/v2.24.1/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
   ```

2. **使二进制文件可执行**：
   
   ```bash
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **测试安装**：
   
   ```bash
   docker-compose -v
   ```
   
   确认输出的版本号满足您的要求。

完成以上步骤后，您将拥有满足版本要求的 Docker 和 Docker Compose。如果遇到任何问题，可能需要根据错误消息调整某些命令。

## 三、安装NVIDIA Container Toolkit

### 步骤 1：检查 NVIDIA Docker Toolkit 是否已安装

1. **检查 NVIDIA Container Runtime**：
   
   ```bash
   docker info | grep nvidia
   ```
   
   如果已经安装了 NVIDIA Container Toolkit，你应该能在输出中看到有关 NVIDIA 的信息。如果没有找到，表示可能尚未安装。

### 步骤 2：安装 NVIDIA Container Toolkit

如果 NVIDIA Container Toolkit 尚未安装，你可以按照以下步骤进行安装：

1. **配置 NVIDIA Docker 仓库**：
   
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **更新软件包列表**：
   
   ```bash
   sudo apt-get update
   ```

3. **安装 NVIDIA Docker**：
   
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   ```

4. **配置并重启 Docker 服务**：
   
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

### 步骤 3：测试 NVIDIA Docker 安装

完成安装后，你可以运行一个带有 GPU 支持的测试容器，以确保一切工作正常：

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

这个命令会使用 NVIDIA 的 CUDA 基础镜像来运行 `nvidia-smi` 命令，如果安装正确，它将列出你系统中的 NVIDIA GPU。

## 四、安装git和git-lfs

要检查 Git 和 Git Large File Storage (Git LFS) 是否已安装，并在需要时安装它们，你可以按照以下步骤操作：

### 检查 Git 是否已安装

1. **检查 Git 版本**：
   打开终端，然后输入以下命令来检查 Git 是否已安装，以及其版本：
   
   ```bash
   git --version
   ```
   
   如果这个命令返回了一个版本号，比如 `git version 2.25.1`，这意味着 Git 已经安装在你的系统上。如果终端显示消息说找不到 `git` 命令，那么你需要安装 Git。

### 安装 Git

如果 Git 未安装，你可以使用你的系统的包管理器来安装：

```bash
sudo apt update
sudo apt install git
```

### 检查 Git LFS 是否已安装

1. **检查 Git LFS 版本**：
   类似地，要检查 Git LFS 是否已安装，运行：
   
   ```bash
   git lfs version
   ```
   
   如果这个命令返回了一个版本号，那么 Git LFS 已经安装在你的系统上。如果终端显示消息说找不到 `git lfs` 命令，则需要安装 Git LFS。

### 安装 Git LFS

如果 Git LFS 未安装，你可以按照以下步骤来安装：

1. **下载并安装 Git LFS**：
   
   ```bash
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   ```

2. **设置 Git LFS**：
   安装完 Git LFS 后，你需要运行一次 `git lfs install` 来设置它：
   
   ```bash
   git lfs install
   ```

## 五、安装QAnything

### 下载本项目并执行

```bash
git clone https://github.com/netease-youdao/QAnything.git
cd QAnything
bash run.sh  # 默认在0号GPU上启动，要求30系列，或40系列或A系列显卡且显存大于等于24GB
# 启动后根据实际情况选择输入remote（在云服务器上启动），或local（在本地机器上启动）
# 随后视情况输入ip地址
```

### 如果提示不支持默认后端，可以按如下步骤执行

#### 显存大于等于22GB

```bash
cd /path/to/QAnything/assets/custom_models
git clone https://www.modelscope.cn/models/netease-youdao/Qwen-7B-QAnything
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b hf -m Qwen-7B-QAnything -t qwen-7b-qanything
```

#### 显存大于等于16GB小于22GB，建议使用3B大模型，以下是使用MiniChat-2-3B的示例

```bash
cd /path/to/QAnything/assets/custom_models
git clone https://www.modelscope.cn/netease-youdao/MiniChat-2-3B.git
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b hf -m MiniChat-2-3B -t minichat
```

#### 显存小于16GB可尝试使用1.8B大模型或直接使用openai api

```bash
# 使用openai api
bash ./run.sh -c cloud -i 0 -b default
# 根据提示输入api-key和api-base等参数
```

### 启动成功后可在访问前端页面使用

http://{your_host}:8777/qanything/

### 也可以访问后端API使用

https://github.com/netease-youdao/QAnything/blob/master/docs/API.md
