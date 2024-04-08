/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-09 17:37:50
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:41:49
 * @FilePath: /QAnything/front_end/src/router/index.ts
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
});

router.afterEach(() => {
  close();
});
export default router;
