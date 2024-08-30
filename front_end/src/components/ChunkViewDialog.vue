<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 14:08:46
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-06 10:30:12
 * @FilePath: front_end/src/components/ChunkViewDialog.vue
 * @Description: 上传文件解析结果切片的弹窗
 -->
<template>
  <Teleport to="body">
    <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
      <a-modal
        v-model:open="showChunkModel"
        title="切片分析结果"
        centered
        width="100%"
        wrap-class-name="chunk-modal"
        :destroy-on-close="true"
        :footer="null"
        @cancel="handleCancel"
      >
        <div class="container">
          <div class="file-preview" :style="{ width: filePreviewWidth }" @mousedown="onMouseDown">
            <div class="scale">
              <a-button shape="circle" :icon="h(PlusCircleOutlined)" @click="enlargeHandle" />
              <span class="scale-text">{{ (zoomLevel * 100).toFixed(0) }}%</span>
              <a-button shape="circle" :icon="h(MinusCircleOutlined)" @click="narrowHandle" />
            </div>
            <div class="file-preview-content">
              <Source :zoom-level="zoomLevel" />
            </div>
            <div class="resize-handle"></div>
          </div>
          <div class="chunk-table">
            <div class="export">
              <p class="file-path">file_path: {{ filePath }}</p>
              <a-button type="primary" @click="exportSelected">
                导出选中({{ selectedKeys.size }})
              </a-button>
            </div>
            <a-table
              :columns="columns"
              :data-source="chunkData"
              bordered
              :scroll="{ y: 'calc(100vh - 32px - 40px - 42px - 64px - 56px - 42px - 8px)' }"
              :pagination="paginationConfig"
              :loading="loading"
              :row-selection="{ selectedRowKeys: [...selectedKeys.keys()], onSelect, onSelectAll }"
              @change="changePage"
              @resizeColumn="handleResizeColumn"
            >
              <template #bodyCell="{ column, record }">
                <template v-if="column.dataIndex === 'content'">
                  <HighLightMarkDown
                    :content="
                      editableData[record.key]
                        ? editableData[record.key].editContent
                        : record.content
                    "
                  />
                </template>
                <template v-if="column.dataIndex === 'editContent'">
                  <div>
                    <a-textarea
                      v-if="editableData[record.key]"
                      v-model:value="editableData[record.key][column.dataIndex]"
                      style="margin: -5px 0"
                      show-count
                      auto-size
                    />
                    <div v-else v-html="formattedContent(record.content)" />
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
        </div>
        <div class="footer">
          <a-button type="primary" @click="handleCancel">{{ common.close }}</a-button>
        </div>
      </a-modal>
    </a-config-provider>
  </Teleport>
</template>

<script setup lang="ts">
import { h } from 'vue';
import { PlusCircleOutlined, MinusCircleOutlined } from '@ant-design/icons-vue';
import { useChunkView } from '@/store/useChunkView';
import { getLanguage } from '@/language';
import { downLoad, resultControl } from '@/utils/utils';
import urlResquest from '@/services/urlConfig';
import { message } from 'ant-design-vue';
import { useChatSetting } from '@/store/useChatSetting';
import HighLightMarkDown from '@/components/HighLightMarkDown.vue';
import { useChatSource } from '@/store/useChatSource';
import Source from './Source/index.vue';

const { showChunkModel } = storeToRefs(useChunkView());
const { common } = getLanguage();
const { chatSettingFormActive } = storeToRefs(useChatSetting());

const { setSourceType, setSourceUrl, setTextContent } = useChatSource();

interface IProps {
  kbId: string;
  fileId: string;
  fileName: string;
}

const props = defineProps<IProps>();
const { kbId, fileId, fileName } = toRefs(props);

const columns = ref([
  {
    title: '编号',
    key: 'id',
    dataIndex: 'id',
    width: 70,
    // fixed: 'left',
  },
  {
    title: 'markdown预览',
    key: 'content',
    dataIndex: 'content',
    resizable: true,
    width: 150,
    minWidth: 150,
  },
  {
    title: '分析结果',
    key: 'editContent',
    dataIndex: 'editContent',
    width: 150,
    // resizable: true,
    // minWidth: 150,
  },
  {
    title: '操作',
    key: 'operation',
    dataIndex: 'operation',
    width: 100,
    fixed: 'right',
  },
]);

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

// 多选
const selectedKeys = ref<Map<string, string>>(new Map());

// 点击多选框
const onSelect = (selectedRow: any) => {
  const key = selectedRow.key;
  const content = selectedRow.content;
  if (selectedKeys.value.has(key)) {
    selectedKeys.value.delete(key);
  } else {
    selectedKeys.value.set(key, content);
  }
};

// 点击全选框
const onSelectAll = (...args) => {
  const changeRows = args[2];
  changeRows.map(item => {
    const key = item.key;
    const content = item.content;
    if (selectedKeys.value.has(key)) {
      selectedKeys.value.delete(key);
    } else {
      selectedKeys.value.set(key, content);
    }
  });
};

// 导出按钮
const exportSelected = () => {
  let str = '';
  const contentArr = [...selectedKeys.value.values()];
  contentArr.map(item => {
    str += item + '\n\n';
  });
  beforeDownload(str);
};

const beforeDownload = content => {
  try {
    const fileName = 'output.txt';
    const resFile = new File([content], fileName, { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(resFile);
    downLoad(url, fileName);
    selectedKeys.value.clear();
  } catch (e) {
    message.error('download error');
  }
};

// 整体加载的loading
const loading = ref(false);
// 保存按钮的loading
const isShowLoading = ref(false);

// 分页参数
const paginationConfig = ref({
  pageNum: 1, // 当前页码
  pageSize: 5, // 每页条数
  total: 0, // 数据总数
  showSizeChanger: false,
  showTotal: total => `共 ${total} 条`,
});

// 分页点击
const changePage = pagination => {
  paginationConfig.value.pageNum = pagination.current;
  getChunks(kbId.value, fileId.value);
};

// 数据
const chunkData = ref<IChunkData[]>([]);
const editableData = ref<Record<string, IChunkData>>({});
const chunkId = ref(1);
const filePath = ref('');

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

// 预览的放大缩小
const zoomLevel = ref(1);
const enlargeHandle = () => {
  zoomLevel.value += 0.1;
};
const narrowHandle = () => {
  // 0.15方便精度计算
  if (zoomLevel.value <= 0.15) return;
  zoomLevel.value -= 0.1;
};

// 获取切片
const getChunks = async (kbId: string, fileId: string) => {
  loading.value = true;
  try {
    const res = (await resultControl(
      await urlResquest.getDocCompleted({
        kb_id: kbId,
        file_id: fileId,
        page_id: paginationConfig.value.pageNum,
        page_limit: paginationConfig.value.pageSize,
      })
    )) as any;
    chunkId.value = (paginationConfig.value.pageNum - 1) * paginationConfig.value.pageSize + 1;
    paginationConfig.value.total = res.total_count;
    chunkData.value = [];
    filePath.value = res.file_path;
    res.chunks.forEach((item: any) => {
      chunkData.value.push({
        key: item.chunk_id,
        id: chunkId.value++,
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
      getChunks(kbId.value, fileId.value);
      handleChatSource({ file_id: fileId.value, file_name: fileName.value });
    } else if (!showChunkModel.value) {
      zoomLevel.value = 1;
      chunkData.value = [];
      chunkId.value = 1;
      paginationConfig.value.pageNum = 1;
      paginationConfig.value.total = 1;
      setSourceUrl('');
      setTextContent('');
      selectedKeys.value.clear();
    }
  }
);

// 检查信息来源的文件是否支持窗口化渲染
let supportSourceTypes = ['pdf', 'docx', 'xlsx', 'txt', 'md', 'jpg', 'png', 'jpeg', 'csv', 'eml'];
const checkFileType = filename => {
  if (!filename) {
    return false;
  }
  const arr = filename.split('.');
  if (arr.length) {
    const suffix = arr.pop();
    if (supportSourceTypes.includes(suffix)) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
};

const handleChatSource = file => {
  console.log('handleChatSource', file);
  const isSupport = checkFileType(file.file_name);
  if (isSupport) {
    queryFile(file);
  }
};

async function queryFile(file) {
  try {
    setSourceUrl(null);
    const res: any = await resultControl(await urlResquest.getFile({ file_id: file.file_id }));
    const suffix = file.file_name.split('.').pop();
    const b64Type = getB64Type(suffix);
    console.log('b64Type', b64Type);
    setSourceType(suffix);
    setSourceUrl(`data:${b64Type};base64,${res.file_base64}`);
    if (suffix === 'txt' || suffix === 'md' || suffix === 'csv' || suffix === 'eml') {
      const decodedTxt = atob(res.file_base64);
      const correctStr = decodeURIComponent(escape(decodedTxt));
      setTextContent(correctStr);
    }
  } catch (e) {
    message.error(e.msg || '获取文件失败');
  }
}

let b64Types = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'text/plain',
  'text/markdown',
  'image/jpeg',
  'image/png',
  'image/jpeg',
  'text/csv',
  'message/rfc822',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
];

function getB64Type(suffix) {
  const index = supportSourceTypes.indexOf(suffix);
  return b64Types[index];
}

// 分析结果的换行
const formattedContent = str => {
  return str.replace(/\n/g, '<br />');
};

// 表格内的拖动
// 计算 分析结果 部分是否可以再缩小了
const isColstart2Narrow = () => {
  const minWidth = 150;
  const tableDom = document.querySelector('.chunk-table');
  const colstart2 = tableDom.querySelector('.ant-table-cell[colstart="2"]');
  const currentWidth = colstart2.clientWidth;
  return currentWidth > minWidth;
};

const handleResizeColumn = (w, col) => {
  const direction = w < col.width ? 'left' : 'right';
  if (isColstart2Narrow() || direction === 'left') {
    col.width = w;
  }
};

// 预览拖动相关
const isResizing = ref(false);
const initialX = ref(0);
const initialWidth = ref(0);
const filePreviewWidth = ref('40%');

const setNewWidth = newWidth => {
  const widthRules = {
    min: 300,
  };
  // 最小值
  if (newWidth >= widthRules.min) {
    filePreviewWidth.value = `${newWidth}px`;
  }
};

const onMouseDown = event => {
  console.log(event.target.classList);
  // 检查鼠标按下的元素是否是拖动手柄
  if (event.target.classList.contains('resize-handle')) {
    isResizing.value = true;
    initialX.value = event.clientX;
    initialWidth.value = event.target.parentElement.offsetWidth;

    // 监听全局的鼠标移动和松开事件
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }
};

const onMouseMove = event => {
  if (isResizing.value) {
    const deltaX = event.clientX - initialX.value;
    const newWidth = initialWidth.value + deltaX;
    // 设置新的宽度
    setNewWidth(newWidth);
  }
};

const onMouseUp = () => {
  if (isResizing.value) {
    isResizing.value = false;
    // 移除监听器
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);
  }
};
</script>

<style lang="scss" scoped>
.container {
  display: flex;
  width: 100%;
  height: calc(100% - 42px - 20px);

  .file-preview {
    position: relative;
    width: 40%;
    height: 100%;
    display: flex;
    flex-direction: column;
    //padding: 0 10px;

    .scale {
      display: flex;
      justify-content: flex-end;
      border-radius: 18px;

      .scale-text {
        width: 40px;
        height: 32px;
        padding: 0 5px;
        display: flex;
        justify-content: center;
        align-items: center;
      }
    }

    .file-preview-content {
      position: relative;
      width: 100%;
      height: 100%;
      border: 1px #d9d9d9 solid;
      padding-right: 0;
      padding-bottom: 8px;
      //border-right: 0;
      border-radius: 12px;
      overflow-x: auto;
      box-sizing: border-box;

      &::-webkit-scrollbar {
        //width: 0;
        height: 12px;
      }
    }
  }

  .resize-handle {
    position: absolute !important;
    //top: 50%;
    bottom: 0;
    right: 0 !important;
    width: 1px;
    height: calc(100% - 44px);
    transform: translateY(-6px);
    cursor: ew-resize;
    background-color: #eee;
    user-select: none;
    z-index: 999;

    &:hover {
      width: 2px;
      background-color: #ccc;
    }
  }

  .chunk-table {
    flex: 1;
    min-width: 600px;
    height: 100%;
    margin-left: 10px;
    overflow: hidden;

    .export {
      width: 100%;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;

      .file-path {
        padding: 0 10px;
        font-size: 10px;
        font-weight: 400;
        color: #999;
        word-break: break-all;
      }
    }

    :deep(.ant-table-cell) {
      vertical-align: top;
    }

    :deep(.ant-table-body) {
      &::-webkit-scrollbar {
        height: 12px;
      }
    }

    :deep(.ant-table-wrapper),
    :deep(.ant-spin-nested-loading),
    :deep(.ant-spin-container) {
      height: 100%;
    }

    :deep(.ant-table) {
      min-height: calc(100% - 64px - 42px);
    }

    :deep(.ant-table-cell[colstart='2']) {
      min-width: 150px !important;
    }
  }
}

.footer {
  width: 100%;
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.editable-row-operations {
  width: 100%;
  height: 100%;

  .operation-div {
    display: flex;
    justify-content: space-evenly;
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
