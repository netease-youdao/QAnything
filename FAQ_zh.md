## 运行时报错，比如大模型不回答问题，或上传文件报错，或多个文件解析失败
- 请查看内存使用情况，目前出错大概率是因为内存不足导致的，可以先尝试释放其他程序占用内存，再尝试运行
  - 如果还是不行，可以尝试关闭部分依赖服务，比如ocr服务（只用于jpg，jpeg，png格式解析），再尝试运行
- 请查看README中的[DEBUG](https://github.com/netease-youdao/QAnything/blob/qanything-v2/README_zh.md#debug)，日志中会有详细的报错信息，可以根据报错信息进行调试

## 使用ollama本地服务时报错：Connection error.
- 原因有2:
  - ollama默认运行在127.0.0.1:11434端口上，而在容器内是无法访问到宿主机的127.0.0.1端口的，所以需要将ollama服务绑定到0.0.0.0:11434端口上，详情可以参考[ollama issue 3581](https://github.com/ollama/ollama/issues/3581)以及[ollama FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)
  - docker-compose-xxx.yaml使用yaml内创建的子网QAnything来连接mysql等各个容器，未配置network_mode: host，因此无法访问到宿主机的localhost（0.0.0.0）服务
- 解决方式：（更新最新代码即可，无需任何手动修改）
  - 在macos和windows下，我不再使用子网QAnything，将mysql，milvus，es等端口直接映射到宿主机中，并使用host.docker.internal来自动替换api_base中的localhost，它允许容器访问宿主机的本地服务
  - 在Linux下，host.docker.internal并不是一个内置的功能，因此我在docker-compose-xxx.yaml中添加了network_mode: host，这样容器内部可以直接访问宿主机的localhost服务

## 使用ollama本地服务时问答效果不佳：
- 原因：ollama服务内置上下文长度为2048，且无法通过传参的方式修改（因此使用ollama时，总Token数量设置无法生效），导致相关信息被截断，从而影响问答效果，详情可参考[ollama issue 5902](https://github.com/ollama/ollama/issues/5902)以及[ollama FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-can-i-specify-the-context-window-size)
- 解决方式：（需手动修改）
  - 举例：如果想把qwen2:72b-instruct的上下文从2048修改32000，需要执行如下操作：
```bash
ollama pull qwen2:72b-instruct

ollama show --modelfile qwen2:72b-instruct > Modelfile

vim Modelfile  // 加一行：PARAMETER num_ctx 32000

ollama create -f Modelfile qwen2:72b_ctx32k
```
  - 修改完成后，重启ollama本地服务，并在前端模型配置中将ollama服务的【模型名称】修改为【qwen2:72b_ctx32k】，即可生效