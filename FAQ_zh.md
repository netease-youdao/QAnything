## 在windows上执行docker-compose命令启动时报错：/bin/bash^M: bad interpreter: No such file or directory
- 原因：在windows下创建编辑的shell脚本是dos格式的，而linux却是只能执行格式为unix格式的脚本，所以在windows上编辑过的文件在linux上(windows下执行wsl后的环境通常也是linux)执行时会报错。
- 解决方案：将回车符替换为空字符串
```shell
# 通过命令查看脚本文件是dos格式还是unix格式，dos格式的文件行尾为^M$ ，unix格式的文件行尾为$：
cat -A scripts/run_for_local.sh  # 验证文件格式
sed -i "s/\r//" scripts/run_for_local.sh
sed -i "s/^M//" scripts/run_for_local.sh
cat -A scripts/run_for_local.sh  # 验证文件格式
```

## 在前端页面输入问题后，返回结果报错：Triton Inference Error (error_code: 4)
- 原因：显存不够了，目前在问答过程中大模型和paddleocr占用的显存会逐渐上升且不释放，可能造成显存不够。
- 解决方案：重启服务，优化显存的工作已在计划中
- 原因2：如果发现显存够用，那是因为新版模型与部分显卡型号不兼容。
- 解决方案：请更换为兼容模型和镜像，手动下载模型文件解压并替换models目录，然后重启服务即可。 
    - 将docker-compose-xxx.yaml中的freeren/qanyxxx:v1.0.9改为freeren/qanyxxx:v1.0.8
    - git clone https://www.wisemodel.cn/Netease_Youdao/qanything.git 
    - cd qanything
    - git reset --hard 79b3da3bbb35406f0b2da3acfcdb4c96c2837faf
    - unzip models.zip
    - 替换掉现有的models目录
    - echo "v2.1.0" > models/version.txt  # 手动避过版本检查

## 在前端页面输入问题后，返回结果是类似后面的乱码：omiteatures贶.scrollHeight㎜eaturesodo Curse.streaming pulumi窟IDI贶沤贶.scrollHeight贶贶贶eatures谜.scrollHeight她是
- 原因：显卡型号不支持，例如V100，请使用3080，3090，4080，4090等显卡，显存需要大于16G

## 服务启动报错，在api.log中显示：mysql.connector.errors.DatabaseError: 2003 (HY000): Can't connect to MySQL server on 'mysql-container-local:3306' (111)
- 原因：将之前的QAnything代码拉取下来后，复制了一份代码到其他的地址，其中有一个volumes是mivlus和mysql默认的本地数据卷，复制后可能导致了mysql的数据卷冲突，导致mysql无法启动。
- 解决方案：删除冲突的数据卷volumes，重新启动服务

## 服务启动报错：ERROR: for qanything-container-local Cannot start service qanything_local: could not select device driver "nvidia" with capabilities: [[gpu]]
- 原因：查看nvidia显卡驱动版本是否满足要求，windows下建议直接更新到最新版；另外检查下是否安装了NVIDIA Container Toolkit, windows下需要进入wsl2环境，再参考linux下安装方法：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## 服务启动报错：nvidia-container-cli: mount error: file creation failed: /var/lib/docker/overlay2/xxxxxx/libnvidia-ml.s0.1: file exists: unknown
- 原因：在windows系统上使用docker-compose-linux.yaml启动
- 解决方案：使用docker-compose-windows.yaml启动


