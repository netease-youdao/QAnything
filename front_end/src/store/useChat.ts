/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:28
 * @FilePath: /QAnything/front_end/src/store/useChat.ts
 * @Description:
 */

import { IChatItem } from '@/utils/types';

export const useChat = defineStore(
  'useChat',
  () => {
    //对话列表
    const QA_List = ref<Array<IChatItem>>([]);
    const clearQAList = () => {
      QA_List.value = [];
    };

    const showModal = ref(false);

    return {
      QA_List,
      clearQAList,
      showModal,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
