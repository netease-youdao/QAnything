<!--
 * @Author: zhangxx03 zhangxx03@rd.netease.com
 * @Date: 2023-02-08 14:13:33
 * @LastEditors: zhangxx03 zhangxx03@rd.netease.com
 * @LastEditTime: 2023-05-30 11:37:44
 * @FilePath: /ai-demo/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# ai-demo
> vue(3.2) + vite + vuex + vur router + ts + antd vue
> prettier+eslint+husky+commitlint
> ai部门前端的demo合集

# 开发部署相关流程介绍

## 开发新demo步骤
+ 从master分支拉取对应的代码,命名为：demo名称/v版本名称
+ packaga.json里面的tagname和version会根据分支名称进行修改，所以一定一定要按照上面的格式创建分支
+ 发布测试环境或者线上环境时时会根据packaga.json里面的tagname名称打包部署到相应的目录，确保不同demo之间不会被影响
+ 接口调试直接配置对应的proxy代理即可
  
## 发布流程
+ 测试环境切换到 test-demo名称 分支，合并开发代码即可自动发布
+ 正式环境切换到 master-demo名称 分支，合并开发代码即可自动发布

## 访问地址
+ 测试环境 http://aiweb.inner.youdao.com/[demo名称]/
+ 正式环境 http://aiweb.youdao.com/[demo名称]/

## 开发流程
+ 给localhost添加配置：local.youdao.com
+ yarn install
+ yarn run dev

### 内置功能

## 项目内功能

# 公共组件
  select选择框
  tree
  基础page页面，包含：进入页面loading展示，预加载图片
  上传文件
  上传进度
  二次确认弹框
# axios
  拦截器，包含：请求失败提示，loading
# 样式
  mixins
  公共变量

## 其它
+ 自动生成代码，包含：组件，页面（自动生成路由），axios拦截器等
+ 
