export const useHomeChat = defineStore('useHomeChat', () => {
  // 当前对话问答列表
  const QA_List = ref([]);
  const setQaList = value => {
    QA_List.value = value;
  };

  // 对话列表
  const chatList = ref([]);
  const setChatList = value => {
    chatList.value = value;
  };

  // 当前对话的Id
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
    chatList,
    setChatList,
    chatId,
    setChatId,
    pageId,
    setPageId,
    qaPageId,
    setQaPageId,
  };
});
