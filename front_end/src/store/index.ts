/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:15
 * @FilePath: /QAnything/front_end/src/store/index.ts
 * @Description:
 */
/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:14
 * @FilePath: /QAnything/front_end/src/store/index.ts
 * @Description:
 */

import { createPinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';

// 创建
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
// 导出
export default pinia;
