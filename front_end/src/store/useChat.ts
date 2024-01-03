/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-22 11:39:18
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-29 16:14:29
 * @FilePath: /qanything-open-source/src/store/useChat.ts
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
