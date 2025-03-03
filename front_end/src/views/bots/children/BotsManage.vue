<template>
  <div class="container">
    <div class="bots-manage">
      <div v-if="isLoading" class="loading">
        <a-spin :indicator="indicator" />
      </div>
      <div v-else>
        <BotList v-if="botList.length" />
        <BotsHome v-else />
      </div>
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
const { setBotList } = useBots();
const isLoading = ref(true);

const indicator = h(LoadingOutlined, {
  style: {
    fontSize: '48px',
  },
  spin: true,
});

const getBotList = async () => {
  try {
    const res: any = await resultControl(await urlResquest.queryBotInfo());
    renderData(res, res.length, 0, 10);
  } catch (e) {
    message.error(e.msg || '获取Bot列表失败');
  }
  isLoading.value = false;
};

// 时间分片优化
/**
 * 时间分片，分片渲染数据，以提高渲染效率和性能
 * @param data - 待渲染的数据数组
 * @param total - 数据数组的总长度
 * @param pageNum - 当前页码 0开始
 * @param pageSize - 每页显示的数据条数
 */
const renderData = (data: Array<any>, total: number, pageNum: number, pageSize: number) => {
  if (total <= 0) return;

  // total 比 pageSize 少的时候只渲染 total 条数据
  const renderCount = Math.min(total, pageSize);

  requestAnimationFrame(() => {
    const startIdx = pageNum * pageSize;
    const endIdx = startIdx + renderCount;
    const dataList = data.slice(startIdx, endIdx);
    // 插入到 页面数据中
    setBotList([...botList.value, ...dataList]);
    // 递归
    renderData(data, total - renderCount, pageNum + 1, pageSize);
  });
};

onMounted(async () => {
  await getBotList();
});

onUnmounted(() => setBotList([]));
</script>
<style lang="scss" scoped>
.container {
  background-color: #26293b;
}

.bots-manage {
  width: 100%;
  height: calc(100vh - 64px);
  overflow: auto;
  background: #f3f6fd;
  border-radius: 12px 0 0 0;
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
