<template>
  <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
    <div class="container">
      <Head />
      <div class="page">
        <a-segmented
          v-model:value="segmentedIndex"
          class="segmented"
          :options="segmentedData"
          block
          @change="changeIndex"
        >
          <template #label="{ title }"> {{ title }}</template>
        </a-segmented>
        <div class="content">
          <router-view></router-view>
        </div>
      </div>
    </div>
  </a-config-provider>
</template>

<script setup lang="ts">
import Head from '@/components/Head.vue';
import routeController from '@/controller/router';

const route = useRouter();
const { changePage } = routeController();

const segmentedData = ref([
  {
    value: 0,
    title: '统计',
  },
  {
    value: 1,
    title: '明细',
  },
]);
const segmentedIndex = ref(-1);

const indexMap = new Map([
  [0, '/statistics/overview'],
  [1, '/statistics/details'],
]);

const segmentedMap = new Map([
  ['/statistics/overview', 0],
  ['/statistics/details', 1],
]);

const changeIndex = value => {
  changePage(indexMap.get(value));
};

const getUrl = () => {
  return route.currentRoute.value.path;
};

onMounted(() => {
  segmentedIndex.value = segmentedMap.get(getUrl());
});
</script>

<style scoped lang="scss">
.container {
  width: 100%;
  height: 100%;
  background-color: #f3f6fd;
}

.page {
  width: 100%;
  height: calc(100% - 64px);
  padding: 28px;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}

.segmented {
  width: 170px;
  padding: 5px;
  margin-bottom: 20px;
  font-weight: bold;
  background-color: #e4e9f4;

  :deep(.ant-segmented-item-selected) {
    color: $baseColor;
  }
}

.content {
  width: 100%;
  flex: 1;
  padding: 20px 0;
  box-sizing: border-box;
}
</style>
