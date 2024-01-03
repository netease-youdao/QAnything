/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-10-30 17:47:34
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-10-31 14:51:46
 * @FilePath: \ai-demo\src\store\index.ts
 * @Description:
 */
import { createPinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';

// 创建
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
// 导出
export default pinia;
