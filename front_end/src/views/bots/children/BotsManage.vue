<template>
  <div class="bots-manage">
    <div v-if="isLoading" class="loading">
      <a-spin :indicator="indicator" />
    </div>
    <div v-else>
      <BotList v-if="botList.length" @getBotList="getBotList" />
      <BotsHome v-else />
    </div>
  </div>
</template>
<script lang="ts" setup>
import BotsHome from '@/components/Bots/BotsHome.vue';
import BotList from '@/components/Bots/BotList.vue';
import { useBots } from '@/store/useBots';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { LoadingOutlined } from '@ant-design/icons-vue';

const { botList } = storeToRefs(useBots());
const { setDefaultBotList, setBotList } = useBots();
const isLoading = ref(true);

const indicator = h(LoadingOutlined, {
  style: {
    fontSize: '48px',
  },
  spin: true,
});

const getDefaultBots = async () => {
  try {
    const res: any = await resultControl(await urlResquest.defaultBots({}));
    setDefaultBotList(res);
  } catch (e) {
    message.error(e.msg || '获取示例Bot失败');
  }
};
getDefaultBots();

const getBotList = async () => {
  try {
    const res: any = await resultControl(await urlResquest.botList());
    setBotList(res);
  } catch (e) {
    message.error(e.msg || '获取Bot列表失败');
  }
  isLoading.value = false;
};
getBotList();
</script>
<style lang="scss" scoped>
.bots-manage {
  width: 100%;
  height: calc(100vh - 64px);
  overflow: auto;
  background: #f3f6fd;
  font-family: PingFang SC;
  .loading {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
}
</style>
