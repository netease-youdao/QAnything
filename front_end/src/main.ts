/*
 * @Author: zhangxx03 zhangxx03@rd.netease.com
 * @Date: 2023-02-08 14:13:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-11 14:41:28
 * @FilePath: /ai-demo/src/main.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { createApp } from 'vue';
import App from './App.vue';
import router from './router/index';
import pinia from './store/index';
import '@/styles/common/global.scss';
import 'virtual:svg-icons-register';
import SvgIcon from '@/components/SvgIcon.vue';
// import { setCookie } from '@/utils/cookie';

const vueApp = createApp(App);
vueApp.use(pinia).use(router);
vueApp.component('SvgIcon', SvgIcon);
//本地无法登录   在测试环境登录后，把cookie里JSESSIONID_NEW的值复制过来
// const env = import.meta.env.VITE_APP_MODE;
// console.log(env);
// // if (env === 'dev') {
// //   // setCookie('JSESSIONID_NEW', 'ec0238e3-d17d-43e7-9f60-5f5ef9ce039d');
// // }

// 单点登录配置函数调用
// 项目一进来就调用单点登录，成功之后才渲染vue实例，防止在身份验证之前暴露任何vue资源
// 已经是rd登录，直接判断
// initKeycloak(initVue, vueApp);
function initVue() {
  vueApp.mount('#app');
}
initVue();
