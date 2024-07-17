/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:28
 * @FilePath: /QAnything/front_end/src/store/useChat.ts
 * @Description:
 */
export const useChat = defineStore(
  'useChat',
  () => {
    const showModal = ref(false);

    // 模型配置弹窗
    const showSettingModal = ref(false);

    return {
      showModal,
      showSettingModal,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
