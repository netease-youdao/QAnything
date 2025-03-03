<template>
  <Teleport to="body">
    <a-modal
      v-model:open="newBotsVisible"
      :title="bots.createBots"
      centered
      :destroyOnClose="true"
      width="480px"
      wrap-class-name="new-bot-modal"
      :footer="null"
    >
      <div class="new-bots-comp">
        <a-form
          :model="formState"
          name="new_bots"
          class="new-bots-form"
          @finish="onFinish"
          @finishFailed="onFinishFailed"
        >
          <a-form-item name="name" :rules="[{ required: true, message: bots.nameCantEmpty }]">
            <div class="item-title">{{ bots.botName }} <span>*</span></div>
            <a-input
              class="name-input"
              v-model:value="formState.name"
              :placeholder="bots.inputName"
              show-count
              :maxlength="20"
              allow-clear
            />
          </a-form-item>
          <a-form-item name="introduction">
            <div class="item-title">{{ bots.botFunctionIntro }}</div>
            <a-textarea
              class="intro-input"
              v-model:value="formState.introduction"
              :placeholder="bots.introBotFunction"
              show-count
              :maxlength="200"
              :auto-size="{ minRows: 3, maxRows: 3 }"
            />
          </a-form-item>
          <!-- <a-form-item>
            <div class="item-title">{{ bots.preSetBot }}</div>
            <div class="preset-bot-list">
              <div
                :class="[
                  'preset-bot-item',
                  item?.code === selectedPrebot?.code ? 'item-active' : '',
                ]"
                v-for="item in defaultBotList"
                :key="item.name"
                @click="selectDefaultBot(item)"
              >
                {{ item.name }}
              </div>
            </div>
          </a-form-item> -->
          <a-form-item>
            <div class="footer">
              <a-button class="cancel-btn" @click="setNewBotsVisible(false)">
                {{ common.cancel }}
              </a-button>
              <a-button :loading="loading" type="primary" html-type="submit" class="login-form-btn">
                {{ common.confirm2 }}
              </a-button>
            </div>
          </a-form-item>
        </a-form>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import { useBotsChat } from '@/store/useBotsChat';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import routeController from '@/controller/router';
import { getLanguage } from '@/language/index';

const { changePage } = routeController();
const { newBotsVisible } = storeToRefs(useBots());
const { setNewBotsVisible, setCurBot, setTabIndex } = useBots();
const { setQaList } = useBotsChat();
const bots = getLanguage().bots;
const common = getLanguage().common;

interface FormState {
  name: string;
  introduction: string;
}

const loading = ref(false);
const selectedPrebot = ref(null); // 选中的预设机器人
const formState = reactive<FormState>({
  name: '',
  introduction: '',
});

const getBotInfo = async botId => {
  try {
    const res: any = await resultControl(await urlResquest.queryBotInfo({ bot_id: botId }));
    setCurBot(res[0]);
  } catch (e) {
    message.error(e.msg || '获取Bot信息失败');
  }
};

const onFinish = async (values: any) => {
  console.log('Success:', values);
  try {
    const res: any = await resultControl(
      await urlResquest.createBot({
        bot_name: values.name,
        description: values.introduction,
      })
    );
    await getBotInfo(res.bot_id);
    message.success(bots.creationSuccessful);
    setTabIndex(0);
    setQaList([]);
    formState.name = '';
    formState.introduction = '';
    selectedPrebot.value = null;
    changePage(`/bots/${res.bot_id}/edit`);
  } catch (e) {
    message.error(e.msg || '创建失败');
  }
  setNewBotsVisible(false);
};

const onFinishFailed = (errorInfo: any) => {
  console.log('Failed:', errorInfo);
};
</script>
<style lang="scss" scoped>
.new-bots-comp {
  width: 100%;
  height: 100%;
  font-family: PingFang SC;
  .item-title {
    font-size: 14px;
    font-weight: 500;
    color: #222222;
    margin-bottom: 12px;
    span {
      color: #ff0000;
    }
  }
  .preset-bot-list {
    width: 100%;
    display: flex;
    // justify-content: space-between;
    flex-wrap: wrap;
    .preset-bot-item {
      width: 96px;
      height: 32px;
      border-radius: 4px;
      background: #fff;
      box-sizing: border-box;
      border: 1px solid #ededed;
      font-size: 14px;
      color: #666666;
      text-align: center;
      line-height: 32px;
      margin-bottom: 16px;
      margin-right: 16px;
      cursor: pointer;
    }
    .item-active {
      border-color: #5a47e5;
      color: #5a47e5;
    }
  }
  .footer {
    display: flex;
    justify-content: end;
    .cancel-btn {
      // width: 68px;
      height: 32px;
      padding: 0 20px;
      margin-right: 16px;
    }
    .login-form-btn {
      // width: 68px;
      height: 32px;
      padding: 0 20px;
      background: #5a47e5 !important;
    }
  }
}
</style>
