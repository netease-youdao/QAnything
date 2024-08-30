/*
 * @Author: zhangxx03 zhangxx03@rd.netease.com
 * @Date: 2023-05-30 10:55:35
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:41:53
 * @FilePath: /QAnything/front_end/src/router/routes.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { RouteRecordRaw } from 'vue-router';

export const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/layout/index.vue'),
    redirect: '/home',
    children: [
      {
        path: '/home',
        name: 'home',
        component: () => import('@/views/Home.vue'),
        meta: {
          title: '首页',
        },
      },
      {
        path: '/bots',
        name: 'bots',
        component: () => import('@/views/bots/Bots.vue'),
        children: [
          {
            path: '/bots',
            name: 'bots',
            component: () => import('@/views/bots/children/BotsManage.vue'),
            meta: {
              requiresAuth: true,
            },
          },
          {
            path: '/bots/:botId/edit',
            name: 'edit',
            component: () => import('@/views/bots/children/BotEdit.vue'),
            meta: {
              requiresAuth: true,
            },
            children: [
              {
                path: '/bots/:botId/edit',
                name: 'edit',
                component: () => import('@/views/bots/children/EditDetail.vue'),
                meta: {
                  requiresAuth: true,
                },
              },
              {
                path: '/bots/:botId/publish',
                name: 'publish',
                component: () => import('@/views/bots/children/BotPublish.vue'),
                meta: {
                  requiresAuth: true,
                },
              },
            ],
          },
        ],
      },
      {
        path: '/quickstart',
        name: 'quickstart',
        component: () => import('@/views/QuickStart/index.vue'),
      },
    ],
  },
  {
    path: '/statistics',
    name: 'statistics',
    component: () => import('@/views/Statistics/index.vue'),
    redirect: '/statistics/overview',
    children: [
      {
        path: 'overview',
        name: 'overview',
        component: () => import('@/views/Statistics/components/Overview.vue'),
      },
      {
        path: 'details',
        name: 'details',
        component: () => import('@/views/Statistics/components/Details.vue'),
      },
    ],
  },
  {
    path: '/bots/:botId/share',
    name: 'share',
    component: () => import('@/views/bots/children/BotShare.vue'),
  },
  {
    path: '/:catchAll(.*)',
    redirect: '/home',
  },
];
