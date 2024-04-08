<template>
  <div class="bot-share">
    <div v-if="isLoading" class="loading">
      <a-spin :indicator="indicator" />
    </div>
    <ChatShare v-else :bot-info="botInfo" chat-type="share" @bot-init="init" />
  </div>
</template>
<script lang="ts" setup>
import ChatShare from '@/components/Bots/ChatShare.vue';
import { useBotsChat } from '@/store/useBotsChat';
import FingerprintJS from '@fingerprintjs/fingerprintjs';
import urlResquest from '@/services/urlConfig';
import { setShareToken } from '@/utils/token';
import { message } from 'ant-design-vue';
import routeController from '@/controller/router';
import { LoadingOutlined } from '@ant-design/icons-vue';

const { getCurrentRoute } = routeController();
const { QA_List } = storeToRefs(useBotsChat());
const botInfo = ref(null);
const botId = ref(null);
const isLoading = ref(true);

const indicator = h(LoadingOutlined, {
  style: {
    fontSize: '48px',
  },
  spin: true,
});

// 移动端适配 禁止用户缩放分享问答页面
onMounted(() => {
  const meta = document.createElement('meta');
  meta.name = 'viewport';
  meta.content = 'width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover';
  meta.id = 'no-zoom';
  document.getElementsByTagName('head')[0].appendChild(meta);
});

onUnmounted(() => {
  document.title = 'Qanything';
  const meta = document.getElementById('no-zoom');
  if (meta) {
    meta.parentNode.removeChild(meta);
  }
});

init();

const loginVirtualUser = async uuid => {
  try {
    const res: any = await urlResquest.loginVirtualUser({ uuid: uuid });
    if (+res.errorCode === 0) {
      setShareToken(res.result);
    } else {
      setTimeout(() => {
        init();
      }, 5000);
      throw new Error(res);
    }
  } catch (e) {
    console.log(e.msg);
  }
};

const getQaList = async botId => {
  try {
    const res: any = await urlResquest.botQaList({ botId: botId });
    if (+res.errorCode === 0) {
      res.result.forEach(item => {
        addQuestion(item.question);
        addAnswer(item.question, item.answer, item.picList);
      });
      isLoading.value = false;
    } else if (+res.errorCode === 403) {
      init();
    } else {
      throw new Error(res);
    }
  } catch (e) {
    message.error(e.msg || '获取问答历史失败');
  }
};

const getBotInfo = async botId => {
  try {
    console.log('zj-botId', botId);
    const res: any = await urlResquest.queryBotInfo({}, {}, botId);
    if (+res.errorCode === 0) {
      botInfo.value = res.result;
      document.title = `Qanything-${res.result.name}`;
      getQaList(res.result.id);
    } else if (+res.errorCode === 403) {
      init();
    } else {
      throw new Error(res);
    }
  } catch (e) {
    message.error(e.msg || '获取Bot信息失败');
  }
};

async function init() {
  try {
    const route = getCurrentRoute();
    botId.value = route.value.params.botId;
    // 初始化FingerprintJS
    const fp = await FingerprintJS.load();

    // 获取浏览器指纹
    const result = await fp.get();

    await loginVirtualUser(result.visitorId);
    await getBotInfo(botId.value);
    console.log('visitorId', result.visitorId);
  } catch (error) {
    console.error('获取浏览器指纹失败', error);
  }
}

function addQuestion(q) {
  QA_List.value.push({
    question: q,
    type: 'user',
  });
  // scrollBottom();
}

function addAnswer(question: string, answer: string, picList) {
  QA_List.value.push({
    answer,
    question,
    type: 'ai',
    copied: false,
    like: false,
    unlike: false,
    source: [],
    showTools: true,
    picList,
  });
}
</script>
<style lang="scss" scoped>
.bot-share {
  width: 100%;
  height: 100%;
  margin: 0 auto;
  overflow: hidden;
  touch-action: manipulation;
  .loading {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
}
</style>
