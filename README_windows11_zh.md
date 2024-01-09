# windows11 下遇到的一些问题

执行`docker-compose -f docker-compose-windows.yaml up qanything_local`后，
前端服务 `http://localhost:5052/qanything/` 无法访问，可能有以下几个情况：
---
1. 情况1
    ```text
    error during connect: this error may indicate that the docker daemon is not running: 
    Get "http://%2F%2F.%2Fpipe%2Fdocker_engine/v1.24/containers/json?all=1&filters=%7B%22label%22%3A%7B%22com.docker.compose.config-hash%22%3Atrue%2C%22com.docker.compose.project%3Dqanything%22%3Atrue%7D%7D": 
    open //./pipe/docker_engine: The system cannot find the file specified.
    ```
    
    原因：docker未启动
    
    解决办法：启动自己的docker客户端

---

2. 情况二：No such file or directory
```text
/workspace/qanything_local/scripts/run_for_local.sh^M: bad interpreter: No such file or directory
```
原因：在windows上编辑好shell脚本，然后在linux系统中执行时，报错/bin/bash^M: bad interpreter: No such file or directory。

解决方案：https://www.cnblogs.com/xyztank/articles/16594936.html

具体操作如下：
1. 在前面`docker-compose`启动的情况下，进入WSL
    - 打开cmd
    - 执行：`docker exec -it qanything-container-local bash`
    - 进入wsl之后，修改 `run_for_local.sh`，如下：
    ```shell
    cd /workspace/qanything_local/scripts/
    
    # 将dos的回车符替换为空字符串
    sed -i "s/\r//g" run_for_local.sh
    sed -i "s/^M//g" run_for_local.sh
    ```
   
    然后重新执行
    ```shell
    docker-compose -f docker-compose-windows.yaml up qanything_local
    ```
---

### 结束WSL中某个进程

查看进程：
```shell
ps auf
```
显示如下：
```text
root      1890  0.5  0.4 797336 78092 pts/1    Sl+  21:54   0:00  \_ node /usr/local/lib/nodejs/node-v18.19.0-linux-x64/bin/yarn dev
root      1901 49.9  2.5 22178752 420964 pts/1 Dl+  21:54   0:13      \_ /usr/local/lib/nodejs/node-v18.19.0-linux-x64/bin/node /workspace/qanything_local/front_end/node_modules/.bin/vite
root      1920  0.1  0.0 722356 14932 pts/1    Sl+  21:54   0:00          \_ /workspace/qanything_local/front_end/node_modules/esbuild-linux-64/bin/esbuild --service=0.14.54 --ping
```

结束进程, 可能要结束同类型的多个进程
```shell
kill 1890 1901 1920
```