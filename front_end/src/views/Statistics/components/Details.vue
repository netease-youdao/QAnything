<template>
  <div class="details-container">
    <div class="choose-area">
      <div class="left item-flex">
        <a-range-picker
          v-model:value="searchDate"
          :open="flag"
          @focus="pickerFocus"
          @blur="pickerBlur"
          @change="dateChange"
        />
        <a-input
          v-model:value="searchConfig.question"
          placeholder="提问内容"
          style="width: 200px; margin-left: 5px"
        />

        <a-tooltip title="search">
          <a-button
            shape="circle"
            :icon="h(SearchOutlined)"
            style="margin-left: 10px"
            @click="searchHandle"
          />
        </a-tooltip>
      </div>

      <div class="right item-flex">
        <div class="export-part button" @click="exportSelected">导出选中</div>
        <div class="export-all button" @click="exportAll">导出全部</div>
      </div>
    </div>
    <div class="table">
      <a-table
        :data-source="dataSource"
        :columns="columns"
        :pagination="paginationConfig"
        :loading="loading"
        :locale="{ emptyText: home.emptyText }"
        :row-selection="{ selectedRowKeys: state.selectedRowKeys, onChange: onSelectChange }"
        :hide-on-single-page="true"
        :show-size-changer="false"
        @change="onChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'kbIds'">
            <a-tooltip placement="topLeft" color="#fff">
              <template #title>
                <span style="color: #666; user-select: text">{{ record.kbIds }}</span>
              </template>
              <span>{{ record.kbIds }}</span>
            </a-tooltip>
          </template>
          <template v-else-if="column.key === 'question'">
            <a-tooltip placement="topLeft" color="#fff">
              <template #title>
                <span style="color: #666; user-select: text">{{ record.question }}</span>
              </template>
              <span>{{ record.question }}</span>
            </a-tooltip>
          </template>
          <template v-else-if="column.key === 'answer'">
            <a-tooltip placement="topLeft" color="#fff">
              <template #title>
                <span style="color: #666; user-select: text">{{ record.answer }}</span>
              </template>
              <span>{{ record.answer }}</span>
            </a-tooltip>
          </template>
          <template v-else-if="column.key === 'options'">
            <a-popconfirm
              overlay-class-name="del-pop"
              placement="topRight"
              :title="statistics.exportTitle"
              :ok-text="common.confirm"
              :cancel-text="common.cancel"
              @confirm="confirmExportItem(record)"
            >
              <a-button type="link" class="export-item">
                {{ statistics.export }}
              </a-button>
            </a-popconfirm>
          </template>
        </template>
      </a-table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { h } from 'vue';
import urlResquest from '@/services/urlConfig';
import { downLoad, getContentDispositionByHeader, resultControl } from '@/utils/utils';
import { getLanguage } from '@/language';
import { SearchOutlined } from '@ant-design/icons-vue';
import message from 'ant-design-vue/es/message';

const { home, common, statistics } = getLanguage();

interface IQADetails {
  key: string;
  kbIds: string[] | string;
  question: string;
  answer: string;
  date: string;
}

const dataSource = ref<IQADetails[]>([]);

const columns = [
  {
    title: '知识库Id',
    dataIndex: 'kbIds',
    key: 'kbIds',
    width: '10%',
    ellipsis: true,
  },
  {
    title: '提问',
    dataIndex: 'question',
    key: 'question',
    width: '35%',
    ellipsis: true,
  },
  {
    title: '回答',
    dataIndex: 'answer',
    key: 'answer',
    width: '35%',
    ellipsis: true,
  },
  {
    title: '时间',
    dataIndex: 'date',
    key: 'date',
    width: '10%',
  },
  {
    title: home.operate,
    key: 'options',
    width: '5%',
  },
];

// faq的分页参数
const paginationConfig = ref({
  current: 1, // 当前页码
  pageSize: 6, // 每页条数
  total: 0, // 数据总数
  showSizeChanger: false,
  showTotal: total => `共 ${total} 条`,
});

// 切换分页的加载
const loading = ref(false);

const state = reactive<{
  selectedRowKeys: string[];
  loading: boolean;
}>({
  selectedRowKeys: [],
  loading: false,
});

// 点击多选框
const onSelectChange = (selectedRowKeys: string[]) => {
  console.log('selectedRowKeys changed: ', selectedRowKeys);
  state.selectedRowKeys = selectedRowKeys;
};

const searchConfig = ref({
  startDate: '',
  endDate: '',
  question: '',
});

// date需要单独绑定，做处理
const searchDate = ref([]);

const flag = ref(false);

const pickerFocus = () => {
  flag.value = true;
};

const pickerBlur = () => {
  flag.value = false;
};

// 日期更改
const dateChange = (date, dateString) => {
  console.log('date', date);
  searchConfig.value.startDate = dateString[0];
  searchConfig.value.endDate = dateString[1];
};

// 点击搜索
const searchHandle = () => {
  console.log(searchConfig.value);
  const { startDate: time_start, endDate: time_end, question: query } = searchConfig.value;
  getQADetail({ time_start, time_end, query });
};

// 请求list数据
const getQADetail = async (...args) => {
  loading.value = true;
  try {
    const res: any = await resultControl(
      await urlResquest.getQAInfo({
        page_id: paginationConfig.value.current,
        page_limit: paginationConfig.value.pageSize,
        ...args[0],
      })
    );
    dataSource.value = [];
    paginationConfig.value.total = res.total_count;
    const { qa_infos } = res;
    qa_infos.map(item => {
      dataSource.value.push({
        key: item.qa_id,
        kbIds: item.kb_ids.toString().replaceAll(',', `\n`),
        question: item.condense_question,
        answer: item.result,
        date: item.timestamp,
      });
    });
    console.log(dataSource.value);
  } catch (e) {
    message.error(e.msg || common.error);
  } finally {
    loading.value = false;
  }
};

// 切换分页
const onChange = pagination => {
  paginationConfig.value.current = pagination.current;
  const { startDate: time_start, endDate: time_end, question: query } = searchConfig.value;
  getQADetail({ time_start, time_end, query });
};

// 表格中导出按钮
const confirmExportItem = record => {
  console.log(record);
  // TODO 请求excel
  beforeDownload('');
};

// 导出选中按钮
const exportSelected = () => {};

// 导出全部按钮
const exportAll = () => {};

const beforeDownload = res => {
  const fileName = getContentDispositionByHeader(res.headers) || 'examlpe.xlsx';
  const resFile = new File([res.data], fileName, { type: 'application/excel' });
  const url = URL.createObjectURL(resFile);
  downLoad(url, fileName);
};

onMounted(() => {
  getQADetail();
});
</script>

<style scoped lang="scss">
.details-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.choose-area {
  height: 32px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
}

.item-flex {
  display: flex;

  .icon {
    width: 68px;
    height: 32px;
    @include flex-center;
    background: #ffffff;
    border-radius: 6px;
    border: 1px solid #e5e5e5;
    margin-left: 16px;
    font-size: 14px;
    color: #222222;
    cursor: default;

    img {
      width: 20px;
      height: 20px;
    }
  }

  .button {
    width: 88px;
    height: 32px;
    line-height: 22px;
    border-radius: 6px;
    opacity: 1;
    background: #ffffff;
    @include flex-center;
    font-size: 14px;
    color: #666666;
    border: 1px solid #e5e5e5;
    cursor: pointer;

    &.export-all {
      color: #ffffff;
      background: #5a47e5;
      margin-left: 16px;
    }
  }
}

.table {
  overflow: auto;

  .export-item {
    padding-left: 0;
  }
}
</style>
