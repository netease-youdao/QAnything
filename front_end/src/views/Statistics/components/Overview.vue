<template>
  <div class="chart-container">
    <LineEchart v-if="!loading" title="总对话量" format-desc="问答数量" :list="chatQAChartList" />
    <LineEchart v-if="!loading" title="知识库总文件" format-desc="文件数量" :list="kbChartList" />
  </div>
</template>

<script setup lang="ts">
import { resultControl } from '@/utils/utils';
import urlConfig from '@/services/urlConfig';
import LineEchart, { type chartListType } from '@/views/Statistics/components/lineEchart.vue';

type FileStatus = 'green' | 'yellow' | 'red' | 'gray';

interface IKbInfo {
  date: string;
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
    const date = i;
    const fileStatus = infos[i];
    kbInfoData.value.push({ date, fileStatus });
  }
  handleKbChartList(kbInfoData.value);
};

// 处理表格的信息
const handleKbChartList = (kbInfoData: IKbInfo[]) => {
  kbInfoData.map(item => {
    kbChartList.value.push({
      name: item.date,
      value: Object.values(item.fileStatus).reduce((sum, currentValue) => sum + currentValue, 0),
    });
  });
  loading.value = false;
};

// 获取知识库相关信息
const getKbInfo = async () => {
  const res: any = await resultControl(await urlConfig.getKbInfo({ by_date: true }));
  handleKbInfo(res.status.zzp__1234);
};

// 获取对话记录相关信息
const getQAInfo = async () => {
  const res: any = await resultControl(await urlConfig.getQAInfo());
  console.log(res);
};

onMounted(() => {
  getKbInfo();
  getQAInfo();
});
</script>

<style scoped lang="scss">
.chart-container {
  width: 100%;
  height: 100%;
}
</style>
