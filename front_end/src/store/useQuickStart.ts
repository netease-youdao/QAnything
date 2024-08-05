// import { resultControl } from '@/utils/utils';
// import urlResquest from '@/services/urlConfig';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { IChatItem } from '@/utils/types';
import { resultControl } from '@/utils/utils';
import urlResquest from '@/services/urlConfig';

const { currentId, knowledgeBaseList } = storeToRefs(useKnowledgeBase());

/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-22 18:18:48
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-02 17:23:08
 * @FilePath: front_end/src/store/useQuickStart.ts
 * @Description:
 */
export interface IHistoryList {
  historyId: number;
  title: string;
  kbId?: string;
}

interface IChatList {
  historyId: number;
  // list: IChatItem[];
  list: IChatItem[];
}

export const useQuickStart = defineStore(
  'useQuickStart',
  () => {
    // 当前对话问答列表
    const QA_List = ref([]);
    const setQaList = value => {
      QA_List.value = value;
    };

    // 是否加载
    const showLoading = ref(false);

    // 上下文数量
    let contextLength = ref<number>(QA_List.value.length);
    watch(
      () => QA_List.value.length,
      () => {
        contextLength.value = QA_List.value.length;
      }
    );

    // 历史记录列表
    const historyList = ref<IHistoryList[]>([]);
    const setHistoryList = (curHistoryList: IHistoryList[]) => {
      historyList.value = curHistoryList;
    };
    const addHistoryList = (title: string) => {
      const newHistoryId = (historyList.value.at(-1)?.historyId || 0) + 1;
      const newHistory: IHistoryList = {
        historyId: newHistoryId,
        title,
        // kbId: value.kbId,
      };

      const curHistoryList = [...historyList.value, newHistory];
      setHistoryList(curHistoryList);
      return newHistoryId;
    };
    const deleteHistoryList = historyId => {
      const filterHistoryList = historyList.value.filter(item => item.historyId !== historyId);
      setHistoryList(filterHistoryList);
    };
    const updateHistoryList = (title: string, historyId: number, kbId: string) => {
      const curHistoryList = historyList.value.map(item => {
        if (item.historyId === historyId) {
          item.title = title;
          item.kbId = kbId;
        }
        return item;
      });
      setHistoryList(curHistoryList);
    };
    const getHistoryById = (historyId: number) => {
      return historyList.value.find(item => item.historyId === historyId);
    };
    const renameHistory = (historyId: number, title: string) => {
      historyList.value.forEach(async item => {
        if (item.historyId === historyId) {
          await resultControl(
            await urlResquest.kbConfig({ kb_id: kbId.value, new_kb_name: title })
          );
          item.title = title;
        }
      });
    };

    // 总的对话列表，二维数组，有每个数据里面有自己的historyId
    const chatList = ref<IChatList[]>([]);
    const addChatList = (historyId: number, QA_List: any[]) => {
      const newChat: IChatList = {
        historyId,
        list: QA_List,
      };
      const isExist = chatList.value.some(item => item.historyId === historyId);
      if (isExist) {
        chatList.value.forEach(item => {
          if (item.historyId === historyId) {
            item.list = QA_List;
          }
        });
      } else {
        chatList.value.push(newChat);
      }
    };
    const getChatById = (historyId: number) => {
      return chatList.value.find(item => item.historyId === historyId);
    };
    const clearChatList = (historyId: number) => {
      chatList.value = chatList.value.filter(item => item.historyId !== historyId);
      historyList.value = historyList.value.filter(item => item.historyId !== historyId);
      chatId.value = null;
      kbId.value = '';
      QA_List.value = [];
    };

    // 当前对话的Id，对应历史记录的historyId
    const chatId = ref(null);
    const setChatId = value => {
      chatId.value = value;
    };

    // 当前知识库的Id，对应历史记录的kbId
    const kbId = ref('');
    const setKbId = value => {
      kbId.value = value;
    };
    watch(
      () => kbId.value,
      () => {
        currentId.value = kbId.value;
      }
    );

    watch(
      () => knowledgeBaseList.value,
      () => {
        historyList.value = historyList.value.filter(item => {
          // 这里使用 some 方法检查 knowledgeBaseList 中是否存在当前 item 的 kbId
          return knowledgeBaseList.value.some(i => i.kb_id === item.kbId);
        });
      }
    );

    // 将要删除的chatId
    const deleteChatId = ref(null);

    return {
      QA_List,
      setQaList,
      showLoading,
      contextLength,
      historyList,
      addHistoryList,
      deleteHistoryList,
      updateHistoryList,
      getHistoryById,
      renameHistory,
      chatList,
      addChatList,
      getChatById,
      clearChatList,
      chatId,
      setChatId,
      kbId,
      setKbId,
      deleteChatId,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);