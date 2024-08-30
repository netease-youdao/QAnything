<template>
  <div class="chart-container">
    <LineEchart
      v-if="!qaLoading"
      title="30天对话量:"
      format-desc="问答数量"
      :list="chatQAChartList"
    />
    <LineEchart
      v-if="!kbLoading"
      title="7天知识库上传文件情况:"
      format-desc="文件数量"
      :list="kbChartList"
    />
  </div>
</template>

<script setup lang="ts">
import { formatDate, getLastDaysRange, resultControl } from '@/utils/utils';
import urlConfig from '@/services/urlConfig';
import LineEchart, { type IChartList } from '@/views/Statistics/components/lineEchart.vue';
import { message } from 'ant-design-vue';

// 问答图表的处理

const qaLoading = ref(false);

const chatQAChartList = ref<IChartList[]>([]);

const handleQAInfo = (infos: object) => {
  const infosArr = Object.entries(infos);
  chatQAChartList.value.push({
    data: infosArr.map(item => ({
      name: item[0],
      value: item[1],
    })),
  });
};

// 获取对话记录相关信息
const getQAInfo = async () => {
  qaLoading.value = true;
  const { time_start, time_end } = getLastDaysRange(30);
  try {
    const res: any = await resultControl(
      await urlConfig.getQAInfo({
        time_start,
        time_end,
        only_need_count: true,
      })
    );
    handleQAInfo(res.qa_infos_by_day);
  } catch (e) {
    message.error(e.msg || '出错了');
  } finally {
    qaLoading.value = false;
  }
};

// 以下是kb图表的处理

type FileStatus = 'green' | 'yellow' | 'red' | 'gray';

interface IKbInfo {
  date: string;
  fileStatus: { [K in FileStatus]: number };
  fileTypes?: any;
}

// 处理后的知识库信息
const kbInfoData = ref<IKbInfo[]>([]);

// 图表的信息
const kbChartList = ref<IChartList[]>([]);

const kbLoading = ref(true);

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
  const listType = [
    {
      type: 'green',
      color: '#91CC75',
      name: '成功',
    },
    {
      type: 'red',
      color: '#EE6666',
      name: '失败',
    },
  ];
  listType.map(item => {
    kbChartList.value.push({
      options: {
        lineColor: item.color,
        name: item.name,
      },
      data: kbInfoData.map(info => ({
        name: formatDate(info.date, '-'),
        value: info.fileStatus[item.type] || 0,
      })),
    });
  });
};

// 获取知识库相关信息
const getKbInfo = async () => {
  try {
    const res: any = await resultControl(await urlConfig.getKbInfo({ by_date: true }));
    handleKbInfo(res.status.zzp__1234);
  } catch (e) {
    message.error(e.msg || '出错了');
  } finally {
    kbLoading.value = false;
  }
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
