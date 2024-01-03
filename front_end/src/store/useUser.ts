/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-10 14:46:01
 * @FilePath: \ai-demo\src\store\useUser.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { getCookie } from '@/utils/cookie';
export const useUser = defineStore(
  'user',
  () => {
    const userInfo = ref({
      token: getCookie('JSESSIONID_NEW'),
    });

    const setUserInfo = info => {
      userInfo.value = info;
    };

    return {
      userInfo,
      setUserInfo,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
