<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 14:08:46
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-26 18:50:33
 * @FilePath: front_end/src/components/ChunkViewDialog.vue
 * @Description: 上传文件解析结果切片的弹窗
 -->
<template>
  <!--  <Teleport to="body">-->
  <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
    <a-modal
      v-model:open="showChunkModel"
      title="切片分析结果"
      centered
      width="40vw"
      wrap-class-name="chunk-modal"
      :footer="null"
      :destroy-on-close="true"
    >
      <div class="chunk-table">
        <a-table :columns="columns" :data-source="chunkData" bordered :scroll="{ y: '60vh' }">
          <template #bodyCell="{ column, text, record }">
            <template v-if="['content'].includes(column.dataIndex)">
              <div>
                <a-textarea
                  v-if="editableData[record.key]"
                  v-model:value="editableData[record.key][column.dataIndex]"
                  style="margin: -5px 0"
                  auto-size
                />
                <template v-else>
                  {{ text }}
                </template>
              </div>
            </template>
            <template v-else-if="column.dataIndex === 'operation'">
              <div class="editable-row-operations">
                <span v-if="editableData[record.key]">
                  <a-typography-link @click="save(record.key)">Save</a-typography-link>
                  <a @click="cancel(record.key)">Cancel</a>
                </span>
                <a-button v-else type="link" @click="edit(record.key)"> Edit</a-button>
              </div>
            </template>
          </template>
        </a-table>
      </div>
      <div class="footer">
        <a-button type="primary" @click="handleCancel">{{ common.close }}</a-button>
      </div>
    </a-modal>
  </a-config-provider>
  <!--  </Teleport>-->
</template>

<script setup lang="ts">
import { useChunkView } from '@/store/useChunkView';
import { getLanguage } from '@/language';
import { UnwrapRef } from 'vue';
import { resultControl } from '@/utils/utils';
import urlResquest from '@/services/urlConfig';
import { message } from 'ant-design-vue';

const { showChunkModel, chunkKbId, chunkFileId } = storeToRefs(useChunkView());
const { common } = getLanguage();

const columns = [
  {
    title: '切片编号',
    dataIndex: 'id',
    width: '10%',
  },
  {
    title: '分析结果',
    dataIndex: 'content',
  },
  {
    title: '操作',
    dataIndex: 'operation',
    width: '15%',
  },
];

interface IChunkData {
  // 切片唯一标识符
  key: string;
  // 显示的id
  id: number;
  // 内容
  content: string;
}

// 分页参数
const paginationConfig = ref({
  current: 1, // 当前页码
  pageSize: 3, // 每页条数
  total: 0, // 数据总数
  showSizeChanger: false,
  showTotal: total => `共 ${total} 条`,
});

// 数据
const chunkData = ref<IChunkData[]>([]);
const editableData: UnwrapRef<Record<string, IChunkData>> = reactive({});

const edit = (key: string) => {
  editableData[key] = chunkData.value.filter(item => key === item.key)[0];
};
const save = (key: string) => {
  Object.assign(chunkData.value.filter(item => key === item.key)[0], editableData[key]);
  delete editableData[key];
};
const cancel = (key: string) => {
  delete editableData[key];
};

const handleCancel = () => {
  showChunkModel.value = false;
};

// 获取切片
const getChunks = async (kbId: string, fileId: string) => {
  try {
    const res = await resultControl(
      await urlResquest.getDocCompleted({ kb_id: kbId, file_id: fileId })
    );
    console.log(paginationConfig.value);
    console.log(res);
    res.chunks!.forEach((item: any) => {
      chunkData.value.push({
        id: 0,
        key: '',
        content: item.page_content,
      });
    });
  } catch (e) {
    message.error(e.msg || '获取文档解析结果失败');
  }
};

watch(
  () => showChunkModel.value,
  () => {
    getChunks(chunkKbId.value, chunkFileId.value);
  }
);
</script>

<style lang="scss" scoped>
.chunk-modal {
  .chunk-table {
    flex: 1;
    overflow-y: auto;
  }

  .footer {
    width: 100%;
    margin-top: 10px;
    display: flex;
    justify-content: flex-end;
  }
}

.editable-row-operations {
  display: flex;
  justify-content: center;
  align-items: center;

  a {
    margin-right: 8px;
  }
}
</style>
