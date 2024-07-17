import { IChatSetting } from '@/utils/types';
/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:45:28
 * @FilePath: /QAnything/front_end/src/store/useChat.ts
 * @Description:
 */
export const useChat = defineStore(
  'useChat',
  () => {
    const showModal = ref(false);

    // 模型配置弹窗
    const showSettingModal = ref(false);

    // 模型配置参数
    const chatSettingFormBase: IChatSetting = {
      modelType: 'openAI', // 默认openAi
      modelName: '',
      apiKey: '',
      apiBase: '',
      apiModelName: '',
      apiContextLength: 0,
      maxToken: 0,
      context: 0,
      temperature: 0.5,
      top_P: 1,
      capabilities: {
        onlineSearch: false,
        mixedSearch: false,
        onlySearch: false,
      },
    };

    // 配置好的模型，包括 openAi Ollama 自定义
    const chatSettingConfigured = ref<IChatSetting[]>([
      {
        ...chatSettingFormBase,
        modelType: 'openAI',
      },
      {
        ...chatSettingFormBase,
        modelType: 'ollama',
      },
      {
        ...chatSettingFormBase,
        modelType: 'custom',
        customId: 0,
      },
    ]);
    const setChatSettingConfigured = (chatSetting: IChatSetting) => {
      if (chatSetting.modelType === 'custom') {
        const index = chatSettingConfigured.value.findIndex(
          item => item.customId === chatSetting.customId
        );
        if (index !== -1) {
          // 找到相同 id 的自定义，替换
          chatSettingConfigured.value[index] = chatSetting;
        } else {
          // 没有找到相同 id 的自定义，添加
          chatSetting.customId = chatSettingConfigured.value.at(-1).customId++;
          chatSettingConfigured.value.push(chatSetting);
        }
      } else if (chatSetting.modelType === 'openAI') {
        chatSettingConfigured.value[0] = chatSetting;
      } else if (chatSetting.modelType === 'ollama') {
        chatSettingConfigured.value[1] = chatSetting;
      }
    };

    return {
      showModal,
      showSettingModal,
      setChatSettingConfigured,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
