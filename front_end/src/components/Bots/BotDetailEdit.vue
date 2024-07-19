<template>
  <div class="bot-detail-edit-comp">
    <div class="name-avatar">
      <img src="@/assets/bots/bot-avatar.png" alt="avatar" />
      <a-input v-model:value="name" :placeholder="bots.inputBotName" show-count :maxlength="20" />
    </div>
    <div class="title">{{ bots.roleSetting }}</div>
    <a-textarea
      v-model:value="roleSetting"
      class="role-setting-input"
      :auto-size="{ minRows: 7, maxRows: 7 }"
      :rows="7"
    />
    <div :class="['role-setting-length', matches && matches.length > 2000 ? 'over-length' : '']">
      {{ matches ? matches.length : 0 }} / 2000
    </div>
    <div class="title">{{ bots.welcomeMessage }}</div>
    <a-textarea
      v-model:value="welcomeMessage"
      class="greeting-input"
      show-count
      :maxlength="100"
      :placeholder="bots.inputWelcomMsg"
      :auto-size="{ minRows: 6, maxRows: 6 }"
      :rows="6"
    />
    <div class="title">{{ bots.associatedKb }}<span>*</span></div>
    <div v-for="(item, index) in curBot.kb_ids" :key="item" class="knowedge-item knowledge-info">
      <img class="knowledge-icon" src="@/assets/bots/knowledge.png" alt="knowledge" />
      <div class="kb-name">{{ curBot.kb_names[index] }}</div>
      <img
        class="remove-icon"
        src="@/assets/bots/remove.png"
        alt="remove"
        @click="removeKb(item)"
      />
    </div>
    <div class="knowedge-item add-knowledge-content" @click="setSelectKnowledgeVisible(true)">
      <img class="add-knowedge" src="@/assets/bots/add-knowedge.png" alt="icon" />
      {{ bots.clickAssociatedKb }}
    </div>
    <div class="save">
      <a-button class="save-btn" type="primary" @click="saveBotInfo">{{ bots.save }}</a-button>
    </div>
    <div class="title">{{ bots.modelSettingTitle }}</div>
    <ChatSettingForm ref="chatSettingFormRef" :context-length="QA_List.length" />
    <div class="chat-setting-form-footer">
      <a-button type="primary" style="width: auto" @click="handleOk">确认应用</a-button>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';
import ChatSettingForm from '@/components/ChatSettingForm.vue';
import { useChatSetting } from '@/store/useChatSetting';
import { useBotsChat } from '@/store/useBotsChat';

const { curBot, knowledgeList } = storeToRefs(useBots());
const { QA_List } = storeToRefs(useBotsChat());
watchEffect(() => {
  console.log('lenghth0-----', QA_List.value.length);
});
const { setSelectKnowledgeVisible, setCurBot } = useBots();
const { setChatSettingConfigured } = useChatSetting();

const bots = getLanguage().bots;

const name = ref('');
const roleSetting = ref('');
const welcomeMessage = ref('');
const matches: any = computed(() => roleSetting.value.match(/[^a-zA-Z\s]|\p{P}|\w+/g));

onMounted(() => {
  console.log('curBot', curBot.value);
  name.value = curBot.value.bot_name;
  welcomeMessage.value = curBot.value.welcome_message;
  roleSetting.value = curBot.value.prompt_setting;
});

const getBotInfo = async botId => {
  try {
    const res: any = await resultControl(await urlResquest.queryBotInfo({ bot_id: botId }));
    console.log('getBotInfo', res);
    setCurBot(res[0]);
  } catch (e) {
    message.error(e.msg || '获取Bot信息失败');
  }
};

const saveBotInfo = async () => {
  try {
    await resultControl(
      await urlResquest.updateBot({
        bot_id: curBot.value.bot_id,
        bot_name: name.value,
        prompt_setting: roleSetting.value,
        welcome_message: welcomeMessage.value,
      })
    );
    getBotInfo(curBot.value.bot_id);
    message.success(bots.saveSuccessful);
  } catch (e) {
    console.log('error--', e);
    message.error(e.msg || '保存失败，请重试');
  }
};

const removeKb = async data => {
  let kbIds = curBot.value.kb_ids;
  console.log('removeKb', data, kbIds);
  kbIds = kbIds.filter(item => item != data);
  try {
    await resultControl(
      await urlResquest.updateBot({
        bot_id: curBot.value.bot_id,
        kb_ids: kbIds,
      })
    );
    getBotInfo(curBot.value.bot_id);
    knowledgeList.value = knowledgeList.value.map(item => {
      if (item.kb_id === data) {
        item.state = item.state === 0 ? 1 : 0;
      }
      return item;
    });
    message.success(bots.removalSucessful);
  } catch (e) {
    message.error(e.msg || '请求失败');
  }
};

// 模型设置
const chatSettingFormRef = ref('');
const handleOk = async () => {
  // @ts-ignore 这里确定有onCheck，因为暴露出来了
  const checkRes = await chatSettingFormRef.value.onCheck();
  if (!Object.hasOwn(checkRes, 'errorFields')) {
    setChatSettingConfigured(checkRes);
    message.success('应用成功');
  }
};
</script>
<style lang="scss" scoped>
.bot-detail-edit-comp {
  width: calc(58% - 20px);
  height: calc(100% - 22px);
  padding: 26px;
  background: #fff;
  overflow: auto;

  .name-avatar {
    width: 100%;
    height: 56px;
    display: flex;
    align-items: center;

    img {
      width: 56px;
      height: 56px;
      margin-right: 16px;
    }

    .ant-input-affix-wrapper {
      height: 40px;
    }
  }

  .title {
    font-size: 16px;
    font-weight: 500;
    color: #222222;
    margin-top: 24px;
    margin-bottom: 12px;

    span {
      color: #ff0000;
    }
  }

  .save {
    width: 100%;
    margin-top: 22px;
    display: flex;
    justify-content: end;

    .save-btn {
      width: 68px;
      height: 32px;
      font-size: 14px;
      background: #5a47e5 !important;
      margin-top: 5px;
    }
  }

  .model-select {
    height: 40px;
    margin: 24px 0;
    display: flex;
    align-items: center;

    .model-title {
      margin: 0 28px 0 0;
    }

    .model-list {
      border-radius: 8px;
      font-size: 14px;
      display: flex;

      .model-item {
        height: 40px;
        width: 120px;
        text-align: center;
        line-height: 40px;
        box-sizing: border-box;
        border: 1px solid #ededed;
        cursor: pointer;

        &:first-child {
          border-radius: 8px 0 0 8px;
        }

        &:last-child {
          border-radius: 0 8px 8px 0;
        }
      }

      .model-item-active {
        color: #5a47e5;
        border: 1px solid #5a47e5;
        background: #eeecfc;
        font-weight: 500;
      }
    }
  }

  .knowedge-item {
    width: 100%;
    height: 56px;
    border-radius: 8px;
    background: #f9f9fc;
    font-size: 14px;
    padding: 0 20px;
    margin-top: 12px;
    display: flex;
    align-items: center;
  }

  .add-knowledge-content {
    justify-content: center;
    color: #999999;
    cursor: pointer;

    .add-knowedge {
      width: 20px;
      height: 20px;
      margin-right: 8px;
    }
  }

  .knowledge-info {
    display: flex;
    align-items: center;

    .knowledge-icon {
      width: 32px;
      height: 32px;
      margin-right: 12px;
    }

    .kb-name {
      flex-grow: 1;
      color: #222222;
    }

    .remove-icon {
      width: 20px;
      height: 20px;
      cursor: pointer;
    }
  }

  .role-setting-input {
    overflow: auto !important;
  }

  .role-setting-length {
    width: 100%;
    color: rgba(0, 0, 0, 0.45);
    text-align: end;
  }

  .over-length {
    color: #ff0000;
  }

  .chat-setting-form-footer {
    display: flex;
    justify-content: flex-end;
  }
}
</style>
<style lang="scss">
.bot-detail-edit-comp {
  .ant-input {
    color: #666666 !important;
  }
}
</style>
