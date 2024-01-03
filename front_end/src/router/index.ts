/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-09 17:37:50
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-13 18:33:43
 * @FilePath: \ai-demo\src\router\index.ts
 * @Description:
 */
import { createRouter, createWebHashHistory } from 'vue-router';
import { routes } from './routes';
// import { useUser } from '@/store/useUser';
// 导入进度条
import { start, close } from '@/utils/nporgress';

//是否隐藏NavBar

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
router.beforeEach((to, from, next) => {
  start();
  next();
  // const { userInfo } = storeToRefs(useUser());
  // const { token } = userInfo.value;

  // start();
  // if (to.matched.some(match => match.meta.requiresAuth)) {
  //   //需要登录才能访问的页面
  //   if (token) {
  //     if (to.path === '/login') {
  //       next({
  //         path: from.fullPath,
  //         query: from.query,
  //       });
  //     } else {
  //       next();
  //     }
  //   } else {
  //     next({
  //       path: '/login',
  //       query: { redirect: to.fullPath },
  //     });
  //     // window.location.href = `${window.location.origin}/login.s?redirectUrl=${window.location.origin}/qanything/#${to.path}`;
  //   }
  // } else {
  //   if (token && to.path === '/login') {
  //     next({
  //       path: from.fullPath,
  //       query: from.query,
  //     });
  //   } else {
  //     next();
  //   }
  // }
});

router.afterEach(() => {
  close();
});
export default router;
