/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 李浩坤 lihaokun@corp.netease.com
 * @LastEditTime: 2024-07-18 18:00:00
 * @FilePath: /QAnything/front_end/src/store/useChat.ts
 * @Description: 注意，修改这里的配置参数时一定要清除localStorage缓存
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
