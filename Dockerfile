# 使用Ubuntu 20.04作为基础镜像
FROM python:3.10

# 避免在安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新软件包列表
RUN apt-get update
# 安装wget
RUN apt-get install -y wget
# 创建TikToken缓存目录
RUN mkdir /opt/tiktoken_cache
# 下载TikToken模型缓存
ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
RUN wget -O /opt/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) "$TIKTOKEN_URL"
# 设置环境变量指向TikToken缓存目录
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache


# 更新软件包列表并安装必要的依赖
#RUN apt-get update && apt-get install -y \
#    software-properties-common \
#    curl \
#    && add-apt-repository ppa:deadsnakes/ppa \
#    && apt-get update \
#    && apt-get install -y python3.10 python3.10-distutils

# 安装pip
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
#    && python3.10 get-pip.py \
#    && rm get-pip.py

# 设置Python 3.10为默认Python版本
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
#    && update-alternatives --set python3 /usr/bin/python3.10 \
#    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
#    && update-alternatives --set python /usr/bin/python3.10

# 验证Python版本
RUN python --version

# 复制models文件夹到/workspace目录
#COPY models /workspace/models

# 设置工作目录
WORKDIR /workspace

COPY ../../models /workspace/models

# 复制requirements.txt文件到工作目录
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential
# 安装项目依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 可以添加其他需要的命令或配置

# 设置容器启动时执行的命令，这里使用tail保持容器运行
CMD ["tail", "-f", "/dev/null"]

# 容器启动时的默认命令
CMD ["/bin/bash"]
