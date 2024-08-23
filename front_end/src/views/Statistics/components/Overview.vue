<template>
  <div class="statistics-overview">
    <LineEchart v-if="!loading" title="知识库信息" :list="kbChartList" />
  </div>
</template>

<script setup lang="ts">
import { resultControl } from '@/utils/utils';
import urlConfig from '@/services/urlConfig';
import LineEchart, { type chartListType } from '@/views/Statistics/components/lineEchart.vue';

type FileStatus = 'green' | 'yellow' | 'red' | 'gray';

interface IKbInfo {
  kbId: string;
  kbName: string;
  fileStatus: { [K in FileStatus]: number };
  fileTypes?: any;
}

// 处理后的知识库信息
const kbInfoData = ref<IKbInfo[]>([]);

// 表格的信息
const kbChartList = ref<chartListType[]>([]);

const loading = ref(true);

// 处理知识库信息
const handleKbInfo = (infos: any[]) => {
  for (let i in infos) {
    const splitKb = i.split('KB');
    const kbId = 'KB' + splitKb.at(-1);
    const kbName = splitKb.slice(0, -1).join('KB');
    const fileStatus = infos[i];
    kbInfoData.value.push({ kbId, kbName, fileStatus });
  }
  handleKbChartList(kbInfoData.value);
};

// 处理表格的信息
const handleKbChartList = (kbInfoData: IKbInfo[]) => {
  kbInfoData.map(item => {
    kbChartList.value.push({
      name: item.kbName,
      value: Object.values(item.fileStatus).reduce((sum, currentValue) => sum + currentValue, 0),
    });
  });
  loading.value = false;
};

// 获取知识库相关信息
const getKbInfo = async () => {
  const res: any = await resultControl(await urlConfig.getKbInfo());
  handleKbInfo(res.status.zzp__1234);
};

onMounted(() => {
  getKbInfo();
});
</script>

<style scoped lang="scss">
.statistics-overview {
  width: 100%;
  height: 100%;
}
</style>
