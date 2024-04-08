## 访问地址

- 开发环境 http://localhost:5052/qanything/

## 开发环境

推荐 node 版本 18.16.0
查看 node 版本

```
node -v
```
建议修改QAnything/front_end/src/services/urlConfig.ts 中的userId,您可以自定义一个属于自己的userId.(字母数下划线组成，以字母开头)

## 开发流程

安装依赖

```shell
yarn
# 或
npm install
```

启动 web 服务(开发模式)

```shell
yarn dev
# 或
npm run dev
```

- 如果 npm 下载依赖特别慢，可以更换为淘宝镜像，然后在进行 npm install。

```
npm config set registry https://registry.npmmirror.com
```

## 打包发布

```shell
yarn build
# 或
npm run build
```

发布打包生成后的 dist/qanything

## 启动服务

打包后，可以启动静态服务

```shell
yarn serve
# 或
npm run serve
```
