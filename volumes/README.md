此目录为milvus，es，mysql等数据库容器的外部映射地址，可以在docker-compose-xxx.yaml文件中查看映射关系，这里存储的数据是固化的，如果想清空本地数据删除此文件夹即可（容器外删除无权限，需要进入容器内删除），在volumes内执行rm -r *
