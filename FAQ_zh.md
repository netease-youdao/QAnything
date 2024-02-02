## 在windows上执行bash run.sh时报错：/bin/bash^M: bad interpreter: No such file or directory，或'\r': command not found
- 原因：在windows下创建编辑的shell脚本是dos格式的，而linux却是只能执行格式为unix格式的脚本，所以在windows上编辑过的文件在linux上(windows下执行wsl后的环境通常也是linux)执行时会报错。
- 解决方案：将回车符替换为空字符串
```shell
# 通过命令查看脚本文件是dos格式还是unix格式，dos格式的文件行尾为^M$ ，unix格式的文件行尾为$：
# 可通过 cat -A scripts/run_xx.sh  # 验证文件格式
sed -i "s/\r//" scripts/run_for_local_option.sh
sed -i "s/^M//" scripts/run_for_local_option.sh
sed -i "s/\r//" scripts/run_for_cloud_option.sh
sed -i "s/^M//" scripts/run_for_cloud_option.sh
sed -i "s/\r//" scripts/run.sh
sed -i "s/^M//" scripts/run.sh
```
## 在windows 上执行bash run.sh时提示端口占用：Error response from daemon: Ports are not available: exposing port TCP 0.0.0.0:5052 -> 0.0.0.0:0: listen tcp 0.0.0.0:5052: bind: An attempt was made to access a socket in a way forbidden by its access permissions.
- 原因：windows 上5052端口被Hyper-V随机占用
- 验证：在powershell中输入 `netsh int ipv4 show excludedportrange protocol=tcp` 列出的端口中包含5052所在的端口范围
- 解决方案：重新设置tcp动态端口范围，执行下面的命令，然后重启windows
```shell
 netsh int ipv4 set dynamic tcp start=11000 num=10000
```

## 选择Qwen-7B-QAnything大模型启动，在前端页面输入问题后，返回结果报错：Triton Inference Error (error_code: 4)
- 原因1：显存不够了，目前在问答过程中大模型和paddleocr占用的显存会逐渐上升且不释放，可能造成显存溢出。
- 解决方案：重启服务，或换成更小的大模型，比如3B或1.8B或OpenAI API
- 原因2：如果发现显存够用，则是因为当前显卡型号不支持默认的triton后端，需要切换到vllm后端，或者hf后端
- 解决方案如下： 
```shell
# 算力查询：请对照/path/to/QAnything/scripts/gpu_capabilities.json
# 高算力卡（>=8.0）推荐vllm后端：
cd /path/to/QAnything/assets/custom_models
git lfs install
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd - 
bash run.sh -c local -i 0 -b vllm -m Qwen-7B-QAnything -t qwen-7b-qanything -p 1 -r 0.85
# 低算力高显存卡推荐hf后端：
cd assets/custom_models
git lfs install
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd - 
bash run.sh -c local -i 0 -b hf -m Qwen-7B-QAnything -t qwen-7b-qanything
```

## 在前端页面输入问题后，返回结果是类似后面的乱码：omiteatures贶.scrollHeight㎜eaturesodo Curse.streaming pulumi窟IDI贶沤贶.scrollHeight贶贶贶eatures谜.scrollHeight她是
- 原因：显卡型号不支持，例如V100，请使用3080，3090，4080，4090等显卡，显存需要大于16G

## 服务启动报错，在api.log中显示：mysql.connector.errors.DatabaseError: 2003 (HY000): Can't connect to MySQL server on 'mysql-container-local:3306' (111)
- 原因：将之前的QAnything代码拉取下来后，复制了一份代码到其他的地址，其中有一个volumes是mivlus和mysql默认的本地数据卷，复制后可能导致了mysql的数据卷冲突，导致mysql无法启动。
- 解决方案：删除冲突的数据卷volumes，重新启动服务，在容器外可能会提示没有权限删除，请进入容器内再删除volumes

## 服务启动报错：ERROR: for qanything-container-local Cannot start service qanything_local: could not select device driver "nvidia" with capabilities: [[gpu]]
- 原因：查看nvidia显卡驱动版本是否满足要求，windows下建议直接更新到最新版；另外检查下是否安装了NVIDIA Container Toolkit, windows下需要进入wsl2环境，再参考linux下安装方法：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## 执行bash run.sh时报错：The command 'docker-compose' could not be found in this WSL 2 distro. 
- 报错信息：
```Text
The command 'docker-compose' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.
```
- 原因：Docker Desktop 未正确配置，需要手动打开 WSL 集成开关
- 解决方案：如果你希望在 WSL 中使用 Windows 的 Docker Desktop，请确保 Docker Desktop 配置为允许 WSL 2 集成。这可以通过 Docker Desktop 的设置中的“Resources” -> “WSL Integration”部分进行配置。

## 执行run.sh时拉取镜像失败： ⠼ error getting credentials - err: exit status 1, out: `error getting credentials - err: exit status 1, out: `A specified logon session does not exist. It may already have been terminated.``
- 解决方案：尝试手动拉取镜像，如执行：docker pull milvusdb/milvus:v2.3.4，然后再重新执行bash run.sh


## 执行run.sh时报错：Illegal instruction (core dumped)，或OCR服务无法启动
- 原因：cpu不支持avx指令集，而PaddleOCR依赖avx指令集，导致无法启动OCR服务
- 解决方案：可进入容器后执行"cat /proc/cpuinfo | grep -i avx"，如果没有输出，则说明cpu不支持avx指令集，可参考paddle官网的解决办法，在容器内重装paddleocr：https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html#old-version-anchor-5-Choose%20CPU%2FGPU
```
# If you want to install the Paddle package with avx and openblas, you can use the following command to download the wheel package to the local, and then use python3 -m pip install [name].whl to install locally ([name] is the name of the wheel package):
python3 -m pip download paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/openblas/avx/stable.html --no-index --no-deps
```