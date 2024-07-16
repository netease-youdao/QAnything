// import { IChatItem } from '@/utils/types';

interface IHistoryList {
  historyId: number;
  title: string;
  kbIds?: string[];
}

interface IChatList {
  historyId: number;
  // list: IChatItem[];
  list: any[];
}

export const useHomeChat = defineStore('useHomeChat', () => {
  // 当前对话问答列表
  const QA_List = ref([]);
  const setQaList = value => {
    QA_List.value = value;
  };

  // 历史记录列表
  const historyList = ref<IHistoryList[]>(JSON.parse(localStorage.getItem('historyList')) || []);
  const setHistoryList = (curHistoryList: IHistoryList[]) => {
    historyList.value = curHistoryList;
    localStorage.setItem('historyList', JSON.stringify(historyList.value));
  };
  const addHistoryList = (title: string) => {
    const newHistoryId = (historyList.value.at(-1)?.historyId || 0) + 1;
    const newHistory: IHistoryList = {
      historyId: newHistoryId,
      title,
      // kbIds: value.kbIds,
    };

    const curHistoryList = [...historyList.value, newHistory];
    setHistoryList(curHistoryList);
    return newHistoryId;
  };
  const deleteHistoryList = historyId => {
    const filterHistoryList = historyList.value.filter(item => item.historyId !== historyId);
    setHistoryList(filterHistoryList);
  };
  const updateHistoryList = (title: string, historyId: number, kbIds: string[]) => {
    const curHistoryList = historyList.value.map(item => {
      if (item.historyId === historyId) {
        item.title = title;
        item.kbIds = kbIds;
      }
      return item;
    });
    setHistoryList(curHistoryList);
  };

  // 总的对话列表，二维数组，有每个数据里面有自己的historyId
  const chatList = ref<IChatList[]>(JSON.parse(localStorage.getItem('chatList')) || []);
  const setChatList = () => {
    // chatList.value = value;
    localStorage.setItem('chatList', JSON.stringify(chatList.value));
  };
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
    setChatList();
  };
  const getChatById = (historyId: number): IChatList => {
    return chatList.value.filter(item => item.historyId === historyId)[0];
  };

  // 当前对话的Id，对应历史记录的historyId
  const chatId = ref(null);
  const setChatId = value => {
    chatId.value = value;
  };

  // 对话列表pageId
  const pageId = ref(1);
  const setPageId = value => {
    pageId.value = value;
  };

  // 问答列表pageId
  const qaPageId = ref(1);
  const setQaPageId = value => {
    qaPageId.value = value;
  };

  return {
    QA_List,
    setQaList,
    historyList,
    addHistoryList,
    deleteHistoryList,
    updateHistoryList,
    chatList,
    addChatList,
    getChatById,
    chatId,
    setChatId,
    pageId,
    setPageId,
    qaPageId,
    setQaPageId,
  };
});
