ARG TRITON_VERSION=23.05
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
ARG PIP_OPTIONS="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
FROM ${BASE_IMAGE}


# 设置非交互式前端，防止在安装过程中出现交云提示
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y --no-install-recommends \
        autoconf \
        autogen \
        clangd \
        git-lfs \
        libb64-dev \
        libz-dev \
        locales-all \
        mosh \
        openssh-server \
        python3-dev \
        rapidjson-dev \
        sudo \
        unzip \
        zstd \
        lsof \
        netcat \
        net-tools \
        zip \
        libgl1-mesa-glx \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        libxml2-dev \
        libxslt1-dev \
        zsh && \
    apt-get clean && \
    rm -rf /usr/local/cuda-11.8 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install $PIP_OPTIONS --upgrade pip && pip install $PIP_OPTIONS --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com regex==2023.10.3 fire==0.5.0 && \
    pip install $PIP_OPTIONS --no-cache-dir --ignore-installed blinker==1.7.0 && \
    pip install $PIP_OPTIONS --no-cache-dir tqdm==4.66.1 omegaconf==2.3.0 concurrent-log-handler==0.9.25 && \
    pip install $PIP_OPTIONS --no-cache-dir numpy==1.23.4 transformers==4.31.0 tiktoken==0.4.0 kazoo==2.9.0 psutil==5.9.0 sentencepiece==0.1.99 tritonclient[all]==2.31.0 pynvml==11.5.0 gunicorn==21.2.0 uvicorn==0.25.0 && \
    pip install $PIP_OPTIONS --no-cache-dir ipython==8.17.2 sanic==23.6.0 pymilvus==2.3.4 langchain==0.0.351 paddleocr==2.7.0.3 paddlepaddle-gpu==2.5.2 nltk==3.8.1 pypinyin==0.50.0 mysql-connector-python==8.2.0 sanic_ext==23.6.0 && \
    pip install $PIP_OPTIONS --no-cache-dir onnxruntime-gpu==1.16.3 openai==1.6.1 && \
    pip install $PIP_OPTIONS --no-cache-dir unstructured==0.11.6 unstructured[pptx]==0.11.6 unstructured[md]==0.11.6

# Add FT-backend
RUN rm -rf /opt/tritonserver/include && rm -rf /opt/tritonserver/third-party-src && mv /opt/tritonserver/backends/onnxruntime /opt/tritonserver/onnxruntime && rm -rf /opt/tritonserver/backends && mkdir -p /opt/tritonserver/backends && mkdir /model_repos && mv /opt/tritonserver/onnxruntime /opt/tritonserver/backends/onnxruntime
ENV WORKSPACE /workspace
WORKDIR /workspace
COPY nltk_data /root/nltk_data
COPY paddleocr /root/.paddleocr
COPY qa_ensemble /opt/tritonserver/backends/qa_ensemble
# EXPOSE 8288

# 下载Node.js指定版本的压缩包
RUN wget https://nodejs.org/download/release/v18.19.0/node-v18.19.0-linux-x64.tar.gz

# 创建目录用于存放Node.js
RUN mkdir -p /usr/local/lib/nodejs

# 解压Node.js压缩包到指定目录
RUN tar -zxvf node-v18.19.0-linux-x64.tar.gz -C /usr/local/lib/nodejs

# 设置环境变量，将Node.js的bin目录加入到PATH中
ENV PATH="/usr/local/lib/nodejs/node-v18.19.0-linux-x64/bin:${PATH}"

RUN rm /workspace/node-v18.19.0-linux-x64.tar.gz

RUN mkdir /opt/tiktoken_cache
ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
RUN wget -O /opt/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) $TIKTOKEN_URL
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
# 启动nginx
# CMD ["nginx", "-g", "daemon off;"]

RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd -p
