export const useChatSource = defineStore('useChatSource', () => {
  //是否展示信息来源
  const chatSourceVisible = ref(false);
  const setChatSourceVisible = (flag: boolean) => {
    chatSourceVisible.value = flag;
  };

  const sourceUrl = ref(null);
  const setSourceUrl = value => {
    sourceUrl.value = value;
  };

  const textContent = ref('');
  const setTextContent = value => {
    textContent.value = value;
  };

  const sourceType = ref('pdf');
  const setSourceType = value => {
    sourceType.value = value;
  };

  return {
    chatSourceVisible,
    setChatSourceVisible,
    sourceUrl,
    setSourceUrl,
    sourceType,
    setSourceType,
    textContent,
    setTextContent,
  };
});
