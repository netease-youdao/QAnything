import { useRouter } from 'vue-router';

export const useBots = defineStore('useBots', () => {
  const route = useRouter();
  const getRouterName = () => {
    const name = route.currentRoute.value.name;
    console.log('zj-route-name', name);
    return name === 'edit' ? 0 : 1;
  };
  //是否展示创建Bot Modal
  const newBotsVisible = ref(false);
  const setNewBotsVisible = value => {
    newBotsVisible.value = value;
  };
  //是否展示学则知识库Modal
  const selectKnowledgeVisible = ref(false);
  const setSelectKnowledgeVisible = value => {
    selectKnowledgeVisible.value = value;
  };
  //是否展示复制链接Modal
  const copyUrlVisible = ref(false);
  const setCopyUrlVisible = value => {
    copyUrlVisible.value = value;
  };

  const tabIndex = ref(getRouterName());
  const setTabIndex = value => {
    tabIndex.value = value;
  };

  const QA_List = ref([]);
  const setQaList = value => {
    QA_List.value = value;
  };

  const botList = ref([]);
  const setBotList = value => {
    botList.value = value;
  };

  const defaultBotList = ref([]);
  const setDefaultBotList = value => {
    defaultBotList.value = value;
  };

  // 当前正在编辑的bot
  const curBot = ref(null);
  const setCurBot = value => {
    curBot.value = value;
  };

  const knowledgeList = ref([]);
  const setKnowledgeList = value => {
    knowledgeList.value = value;
  };

  const webUrl = ref('');
  const setWebUrl = value => {
    webUrl.value = value;
  };

  return {
    newBotsVisible,
    setNewBotsVisible,
    botList,
    setBotList,
    tabIndex,
    setTabIndex,
    selectKnowledgeVisible,
    setSelectKnowledgeVisible,
    knowledgeList,
    setKnowledgeList,
    webUrl,
    setWebUrl,
    copyUrlVisible,
    setCopyUrlVisible,
    defaultBotList,
    setDefaultBotList,
    curBot,
    setCurBot,
    QA_List,
    setQaList,
  };
});
