import { IChatSetting } from '@/utils/types';
import { getLanguage } from '@/language';

const common = getLanguage().common;

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
      maxToken: 512,
      chunkSize: 800,
      context: 0,
      temperature: 0.5,
      top_P: 1,
      top_K: 30,
      capabilities: {
        networkSearch: false,
        mixedSearch: false,
        onlySearch: false,
        rerank: false,
      },
      active: false,
    };

    // 配置好的模型，包括 openAi Ollama 自定义
    const chatSettingConfigured = ref<IChatSetting[]>([
      {
        ...chatSettingFormBase,
        modelType: 'openAI',
        active: true, // 默认openAi
      },
      {
        ...chatSettingFormBase,
        modelType: 'ollama',
        apiKey: 'ollama',
        apiBase: 'http://localhost:11434/v1',
        apiContextLength: 2048,
        active: false,
      },
      {
        ...chatSettingFormBase,
        modelName: '',
        modelType: common.customModelType,
        customId: 0,
        active: false,
      },
    ]);
    const setChatSettingConfigured = (chatSetting: IChatSetting) => {
      // 先把所有active设置为false;
      chatSettingConfigured.value.forEach(item => {
        item.active = false;
      });
      // 再把当前的active设置为true
      chatSetting.active = true;
      if (chatSetting.modelType === 'openAI') {
        chatSettingConfigured.value[0] = { ...chatSetting };
      } else if (chatSetting.modelType === 'ollama') {
        chatSettingConfigured.value[1] = { ...chatSetting };
      } else {
        // 自定义
        chatSetting.modelType = chatSetting.modelName;
        const index = chatSettingConfigured.value.findIndex(
          item => item.customId === chatSetting.customId
        );
        if (index !== -1 && chatSetting.customId !== 0) {
          // 找到相同 id 的自定义，替换
          chatSettingConfigured.value[index] = { ...chatSetting };
        } else {
          // 没有找到相同 id 的自定义或为第一个，添加
          chatSetting.customId = chatSettingConfigured.value.at(-1).customId + 1;
          chatSettingConfigured.value.push({ ...chatSetting });
        }
      }
    };

    // 当前应用的设置
    const chatSettingFormActive = computed<IChatSetting>(() => {
      return chatSettingConfigured.value.find(item => item.active === true);
    });

    // openAI默认模型配置
    const openAIs = [
      'gpt-3.5-turbo',
      'gpt-3.5-turbo-1106',
      'gpt-3.5-turbo-0125',
      'gpt-3.5-turbo-0613',
      'gpt-3.5-turbo-instruct',
      'gpt-3.5-turbo-16k',
      'gpt-3.5-turbo-16k-0613',
      'gpt-3.5-turbo-0301',
      'gpt-4-0125-preview',
      'gpt-4-1106-preview',
      'gpt-4-1106-version-preview',
      'gpt-4',
      'gpt-4-0314',
      'gpt-4-0613',
      'gpt-4-32k-0314',
      'gpt-4-32k',
      'gpt-4-turbo-2024-04-09',
      'gpt-4-turbo',
      'gpt-4-turbo-preview',
      'gpt-4o',
      'gpt-4o-mini',
      'gpt-4o-2024-08-06',
      'chatgpt-4o-latest',
    ] as const;

    type OpenAIModel = (typeof openAIs)[number];

    type OpenAISettings = {
      apiContextLength: number;
    };

    const openAISettingMap = new Map<OpenAIModel, OpenAISettings>([
      ['gpt-3.5-turbo', { apiContextLength: 16384 }],
      ['gpt-3.5-turbo-1106', { apiContextLength: 16384 }],
      ['gpt-3.5-turbo-0125', { apiContextLength: 16384 }],
      ['gpt-3.5-turbo-0613', { apiContextLength: 4096 }],
      ['gpt-3.5-turbo-instruct', { apiContextLength: 4096 }],
      ['gpt-3.5-turbo-16k', { apiContextLength: 16384 }],
      ['gpt-3.5-turbo-16k-0613', { apiContextLength: 16384 }],
      ['gpt-3.5-turbo-0301', { apiContextLength: 16384 }],
      ['gpt-4-0125-preview', { apiContextLength: 131072 }],
      ['gpt-4-1106-preview', { apiContextLength: 131072 }],
      ['gpt-4-1106-version-preview', { apiContextLength: 131072 }],
      ['gpt-4', { apiContextLength: 8192 }],
      ['gpt-4-0314', { apiContextLength: 8192 }],
      ['gpt-4-0613', { apiContextLength: 8192 }],
      ['gpt-4-32k-0314', { apiContextLength: 32768 }],
      ['gpt-4-32k', { apiContextLength: 32768 }],
      ['gpt-4-turbo-2024-04-09', { apiContextLength: 131072 }],
      ['gpt-4-turbo', { apiContextLength: 131072 }],
      ['gpt-4-turbo-preview', { apiContextLength: 131072 }],
      ['gpt-4o', { apiContextLength: 131072 }],
      ['gpt-4o-mini', { apiContextLength: 131072 }],
      ['gpt-4o-2024-08-06', { apiContextLength: 131072 }],
      ['chatgpt-4o-latest', { apiContextLength: 131072 }],
    ]);

    return {
      setChatSettingConfigured,
      chatSettingConfigured,
      chatSettingFormActive,
      openAISettingMap,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
