/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:47
 * @FilePath: /QAnything/front_end/src/store/useUser.ts
 * @Description:
 */

export const useUser = defineStore(
  'user',
  () => {
    const userInfo = ref({
      token: '',
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
