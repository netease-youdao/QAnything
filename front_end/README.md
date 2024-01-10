

## 访问地址
+ 开发环境 http://localhost:5052/qanything/


## 开发流程

安装依赖
```shell
yarn
# 或
npm install
```

启动web服务(开发模式)
```shell
yarn dev
# 或
npm run dev
```

* 如果npm下载依赖特别慢，可以更换为淘宝镜像，然后在进行npm install。
```
npm config set registry https://registry.npmmirror.com
```

## 打包发布
```shell
yarn build
# 或
npm run build
```

发布打包生成后的dist/qanything

## 启动服务
打包后，可以启动静态服务
```shell
yarn serve
# 或
npm run serve
```

