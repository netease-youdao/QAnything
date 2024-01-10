# windows11 下遇到的一些问题

双击`run_in_windows.bat` 即可运行

或在cmd中执行如下命令，即可运行。
```shell
Start-Process -FilePath ".\run_in_windows.bat" -Wait -NoNewWindow
```

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