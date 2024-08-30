## 运行时报错，比如大模型不回答问题，或上传文件报错，或多个文件解析失败
- 请查看内存使用情况，目前出错大概率是因为内存不足导致的，可以先尝试释放其他程序占用内存，再尝试运行
  - 如果还是不行，可以尝试关闭部分依赖服务，比如ocr服务（只用于jpg，jpeg，png格式解析），再尝试运行
- 请查看README中的[DEBUG](https://github.com/netease-youdao/QAnything/blob/qanything-v2/README_zh.md#debug)，日志中会有详细的报错信息，可以根据报错信息进行调试
