import { IChatSetting } from '@/utils/types';

export const useChatSetting = defineStore(
  'useChatSetting',
  () => {
    // 模型配置参数
    const chatSettingFormBase: IChatSetting = {
      modelType: '',
      apiKey: '',
      apiBase: '',
      apiModelName: '',
      apiContextLength: 4096,
      maxToken: 2048,
      context: 0,
      temperature: 0.5,
      top_P: 1,
      capabilities: {
        onlineSearch: false,
        mixedSearch: false,
        onlySearch: false,
      },
      active: false,
    };

    // 配置好的模型，包括 openAi Ollama 自定义
    const chatSettingConfigured = ref<IChatSetting[]>([
      {
        ...chatSettingFormBase,
        modelType: 'openAI',
        active: false, // 默认openAi
      },
      {
        ...chatSettingFormBase,
        modelType: 'ollama',
        active: true,
      },
      {
        ...chatSettingFormBase,
        modelName: '',
        modelType: '自定义模型配置',
        customId: 0,
        active: false,
      },
    ]);
    const setChatSettingConfigured = (chatSetting: IChatSetting) => {
      console.log('chatSetting-------', chatSetting);
      // 先把所有active设置为false;
      chatSettingConfigured.value.forEach(item => {
        item.active = false;
      });
      // 再把当前的active设置为true
      chatSetting.active = true;
      if (chatSetting.modelType === 'openAI') {
        chatSettingConfigured.value[0] = chatSetting;
      } else if (chatSetting.modelType === 'ollama') {
        chatSettingConfigured.value[1] = chatSetting;
      } else {
        debugger;
        // 自定义
        chatSetting.modelType = chatSetting.modelName;
        const index = chatSettingConfigured.value.findIndex(
          item => item.customId === chatSetting.customId
        );
        if (index !== -1 && chatSetting.customId !== 0) {
          // 找到相同 id 的自定义，替换
          chatSettingConfigured.value[index] = chatSetting;
        } else {
          // 没有找到相同 id 的自定义或为第一个，添加
          chatSetting.customId = chatSettingConfigured.value.at(-1).customId + 1;
          chatSettingConfigured.value.push(chatSetting);
        }
      }
    };

    return {
      setChatSettingConfigured,
      chatSettingConfigured,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
