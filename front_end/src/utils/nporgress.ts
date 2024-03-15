/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:28
 * @FilePath: /QAnything/front_end/src/utils/nporgress.ts
 * @Description:
 */

import NProgress from 'nprogress';
import 'nprogress/nprogress.css';

NProgress.configure({
  easing: 'ease', // 动画方式
  speed: 1000, // 递增进度条的速度r
  showSpinner: false, // 是否显示加载ico
  trickleSpeed: 200, // 自动递增间隔
  minimum: 0.3, // 更改启动时使用的最小百分比
  parent: 'body', //指定进度条的父容器
});

// 开启进度条
export const start = () => {
  NProgress.start();
};

// 关闭进度条
export const close = () => {
  NProgress.done();
};
