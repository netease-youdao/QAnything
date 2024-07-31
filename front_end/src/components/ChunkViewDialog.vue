<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 14:08:46
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-31 19:47:35
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
    >
      <div class="chunk-table">
        <a-table
          :columns="columns"
          :data-source="chunkData"
          bordered
          :scroll="{ y: '60vh' }"
          :pagination="paginationConfig"
          @change="changePage"
        >
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
                <div v-if="editableData[record.key]">
                  <a-typography-link v-if="editableData[record.key]" @click="save(record.key)">
                    {{ common.save }}
                  </a-typography-link>
                  <a-typography-text @click="cancel(record.key)">
                    {{ common.cancel }}
                  </a-typography-text>
                </div>
                <a-typography-link v-else @click="edit(record.key)">
                  {{ common.edit }}
                </a-typography-link>
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

const { showChunkModel } = storeToRefs(useChunkView());
const { common } = getLanguage();

interface IProps {
  kbId: string;
  docId: string;
}

const props = defineProps<IProps>();
const { kbId, docId } = toRefs(props);

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
  // 切片唯一标识符 -> chunk_id
  key: string;
  // 显示的id
  id: number;
  // 内容
  content: string;
}

// 分页参数
const paginationConfig = ref({
  pageNum: 1, // 当前页码
  pageSize: 3, // 每页条数
  total: 0, // 数据总数
  showSizeChanger: false,
  showTotal: total => `共 ${total} 条`,
});

// 分页点击
const changePage = pagination => {
  paginationConfig.value.pageNum = pagination.current;
  getChunks(kbId.value, docId.value);
};

// 数据
const chunkData = ref<IChunkData[]>([]);
const editableData: UnwrapRef<Record<string, IChunkData>> = reactive({});
const chunkId = ref(1);

const edit = (key: string) => {
  editableData[key] = chunkData.value.filter(item => key === item.key)[0];
};

const save = async (key: string) => {
  try {
    await resultControl(
      await urlResquest.updateDocCompleted({
        doc_id: key,
        update_content: editableData[key].content,
      })
    );
    message.success('修改成功');
    Object.assign(chunkData.value.filter(item => key === item.key)[0], editableData[key]);
    delete editableData[key];
  } catch (e) {
    message.error(e.msg || '更新文档解析结果失败');
  }
};

const cancel = (key: string) => {
  delete editableData[key];
};

const handleCancel = () => {
  showChunkModel.value = false;
};

// 获取切片
const getChunks = async (kbId: string, docId: string) => {
  try {
    const res = (await resultControl(
      await urlResquest.getDocCompleted({
        kb_id: kbId,
        file_id: docId,
        page_offset: paginationConfig.value.pageNum,
        page_limit: paginationConfig.value.pageSize,
      })
    )) as any;
    res.chunks.forEach((item: any) => {
      chunkData.value.push({
        id: chunkId.value++,
        key: item.chunk_id,
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
    if (showChunkModel.value) {
      getChunks(kbId.value, docId.value);
    } else {
      chunkData.value = [];
      chunkId.value = 1;
    }
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
  width: 100%;
  height: 100%;

  a {
    margin-right: 8px;
  }
}
</style>
