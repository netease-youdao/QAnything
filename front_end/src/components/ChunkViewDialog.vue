<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 14:08:46
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-06 10:30:12
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
      :destroy-on-close="true"
      :footer="null"
      @cancel="handleCancel"
    >
      <div class="file-preview"></div>
      <div class="chunk-table">
        <a-table
          :columns="columns"
          :data-source="chunkData"
          bordered
          :scroll="{ y: '60vh' }"
          :pagination="paginationConfig"
          :loading="loading"
          @change="changePage"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'content'">
              <HighLightMarkDown
                :content="
                  editableData[record.key] ? editableData[record.key].editContent : record.content
                "
              />
            </template>
            <template v-else-if="column.dataIndex === 'editContent'">
              <div>
                <a-textarea
                  v-if="editableData[record.key]"
                  v-model:value="editableData[record.key][column.dataIndex]"
                  style="margin: -5px 0"
                  show-count
                  auto-size
                />
                <template v-else>
                  {{ record.content }}
                </template>
              </div>
            </template>
            <template v-else-if="column.dataIndex === 'operation'">
              <div class="editable-row-operations">
                <div v-if="editableData[record.key]" class="operation-div">
                  <a-button
                    class="operation-btn"
                    size="small"
                    :loading="isShowLoading"
                    type="link"
                    @click="save(record.key)"
                  >
                    {{ !isShowLoading ? common.save : '' }}
                  </a-button>
                  <a-button
                    v-if="!isShowLoading"
                    class="operation-btn"
                    size="small"
                    :loading="isShowLoading"
                    type="text"
                    @click="cancel(record.key)"
                  >
                    {{ common.cancel }}
                  </a-button>
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
import { Ref } from 'vue';
import { resultControl } from '@/utils/utils';
import urlResquest from '@/services/urlConfig';
import { message } from 'ant-design-vue';
import { useChatSetting } from '@/store/useChatSetting';
import HighLightMarkDown from '@/components/HighLightMarkDown.vue';

const { showChunkModel } = storeToRefs(useChunkView());
const { common } = getLanguage();
const { chatSettingFormActive } = storeToRefs(useChatSetting());

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
    title: 'markdown预览',
    key: 'content',
    dataIndex: 'content',
  },
  {
    title: '分析结果',
    key: 'editContent',
    dataIndex: 'editContent',
  },
  {
    title: '操作',
    dataIndex: 'operation',
    width: '100px',
  },
];

interface IChunkData {
  // 切片唯一标识符 -> chunk_id
  key: string;
  // 显示的id
  id: number;
  // markdown内容
  content: string;
  // 编辑内容
  editContent: string;
}

// 整体加载的loading
const loading = ref(false);
// 保存按钮的loading
const isShowLoading = ref(false);

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
const editableData: Ref<Record<string, IChunkData>> = ref({});
const chunkId = ref(1);

const edit = (key: string) => {
  editableData.value[key] = JSON.parse(
    JSON.stringify(chunkData.value.filter(item => key === item.key)[0])
  );
};

const save = async (key: string) => {
  isShowLoading.value = true;
  message.warn('正在更新……');
  try {
    await resultControl(
      await urlResquest.updateDocCompleted({
        chunk_size: chatSettingFormActive.value.chunkSize,
        doc_id: key,
        update_content: editableData.value[key].editContent,
      })
    );
    message.success('修改成功');
    Object.assign(chunkData.value.filter(item => key === item.key)[0], editableData[key]);
    delete editableData.value[key];
  } catch (e) {
    message.error(e.msg);
  } finally {
    isShowLoading.value = false;
  }
};

const cancel = (key: string) => {
  delete editableData.value[key];
};

const handleCancel = () => {
  showChunkModel.value = false;
  !isShowLoading && (editableData.value = {});
};

// 获取切片
const getChunks = async (kbId: string, docId: string) => {
  loading.value = true;
  try {
    const res = (await resultControl(
      await urlResquest.getDocCompleted({
        kb_id: kbId,
        file_id: docId,
        page_offset: paginationConfig.value.pageNum,
        page_limit: paginationConfig.value.pageSize,
      })
    )) as any;
    chunkId.value = paginationConfig.value.pageSize * paginationConfig.value.pageNum - 2;
    paginationConfig.value.total = res.total_count;
    chunkData.value = [];
    res.chunks.forEach((item: any) => {
      chunkData.value.push({
        id: chunkId.value++,
        key: item.chunk_id,
        content: item.page_content,
        editContent: item.page_content,
      });
    });
  } catch (e) {
    message.error(e.msg || '获取文档解析结果失败');
  } finally {
    loading.value = false;
  }
};

watch(
  () => showChunkModel.value,
  () => {
    if (showChunkModel.value) {
      chunkData.value = [];
      getChunks(kbId.value, docId.value);
    } else if (!showChunkModel.value) {
      chunkData.value = [];
      chunkId.value = 1;
      paginationConfig.value.pageNum = 1;
      paginationConfig.value.total = 1;
    }
  }
);
</script>

<style lang="scss" scoped>
.chunk-modal {
  .chunk-table {
    height: 60vh;
    flex: 1;
    overflow-y: auto;

    :deep(.ant-table-cell) {
      vertical-align: top;
    }
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

  .operation-div {
    display: flex;
    justify-content: space-between;
    align-items: center;

    .operation-btn {
      padding: 0;
    }
  }

  a {
    margin-right: 8px;
  }
}
</style>
