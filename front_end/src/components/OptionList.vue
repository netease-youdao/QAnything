<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-26 14:49:41
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-05 19:10:58
 * @FilePath: /ai-demo/src/components/OptionList.vue
 * @Description: 
-->
<template>
  <div class="list-page">
    <div class="content">
      <div class="options">
        <div class="to-chat" @click="goChat">
          <img src="../assets/home/icon-back.png" alt="back" />
          <span>{{ home.conversation }}</span>
        </div>
        <p class="kb-name">
          <span class="name">{{ currentKbName }}</span>
          <!-- <span class="id">{{ home.knowledgeID }} {{ currentId }}</span> -->
        </p>
      </div>
      <div class="nav-info">
        <div class="navs">
          <div
            :class="['nav-item', navIndex === item.value ? 'nav-item-active' : '']"
            v-for="item in navList"
            :key="item.name"
            @click="navClick(item.value)"
          >
            {{ item.name }}
          </div>
        </div>
        <div class="handle-btn">
          <div v-if="navIndex === 0" class="upload" @click="showFileUpload">{{ home.upload }}</div>
          <div v-if="navIndex === 0" class="add-link" @click="showUrlUpload">{{ home.addUrl }}</div>
          <div v-if="navIndex === 1" class="upload" @click="showEditQaSet">{{ home.inputQa }}</div>
        </div>
      </div>
      <div class="table">
        <a-table
          v-if="navIndex === 0"
          :data-source="dataSource"
          :columns="columns"
          :pagination="false"
          :locale="{ emptyText: home.emptyText }"
        >
          <template #headerCell="{ column }">
            <template v-if="column.key === 'status'">
              <span>{{ home.documentStatus }}</span
              ><span
                class="small"
                style="
                   {
                    font-size: 12px;
                    font-weight: normal;
                    line-height: 18px;
                    color: #44464e;
                  }
                "
                >(解析成功后可问答)</span
              >
            </template>
          </template>

          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'status'">
              <div class="status-box">
                <span class="icon-file-status">
                  <img
                    v-if="record.status === 'gray'"
                    class="loading file-status"
                    src="../assets/home/icon-loading.png"
                    alt="loading"
                  />
                  <SvgIcon
                    v-else
                    class="file-status"
                    :name="record.status === 'green' ? 'success' : 'error'"
                  />
                </span>
                <span> {{ parseStatus(record.status) }}</span>
              </div>
            </template>
            <template v-else-if="column.key === 'options'">
              <a-popconfirm
                overlay-class-name="del-pop"
                placement="topRight"
                :title="common.deleteTitle"
                :ok-text="common.confirm"
                :cancel-text="common.cancel"
                @confirm="confirm"
              >
                <span class="delete-item" @click="deleteItem(record)">{{ common.delete }}</span>
              </a-popconfirm>
            </template>
          </template>
        </a-table>
        <a-table
          v-else
          :data-source="faqList"
          :columns="qaColumns"
          :locale="{ emptyText: home.emptyText }"
          :loading="loading"
          :pagination="paginationConfig"
          @change="onChange"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'status'">
              <div>{{ parseFaqStatus(record.status) }}</div>
            </template>
            <template v-else-if="column.key === 'options'">
              <div class="options">
                <a-button
                  class="edit-item"
                  type="link"
                  @click="editQaItem(record)"
                  :disabled="record.status !== 'green'"
                >
                  {{ bots.edit }}
                </a-button>
                <a-popconfirm
                  overlay-class-name="qa-del-pop"
                  placement="topRight"
                  :title="home.deleteQaSetText"
                  :ok-text="common.confirm"
                  :cancel-text="common.cancel"
                  @confirm="qaConfirm"
                >
                  <a-button class="delete-item" danger type="link" @click="deleteQaItem(record)">
                    {{ common.delete }}
                  </a-button>
                </a-popconfirm>
              </div>
            </template>
          </template>
        </a-table>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import urlResquest from '@/services/urlConfig';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { pageStatus } from '@/utils/enum';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
const { setDefault } = useKnowledgeBase();
const { currentKbName, currentId } = storeToRefs(useKnowledgeBase());
const { setModalVisible, setUrlModalVisible, setModalTitle } = useKnowledgeModal();
import { useOptiionList } from '@/store/useOptiionList';
const { getDetails, setEditQaSet, setEditModalVisible, getFaqList, setFaqType, setPageNum } =
  useOptiionList();
const { dataSource, faqList, timer, faqTimer, total, pageNum, loading } = storeToRefs(
  useOptiionList()
);

import { getLanguage } from '@/language/index';

const home = getLanguage().home;
const common = getLanguage().common;
const bots = getLanguage().bots;

const navIndex = ref(0);

const navList = [
  {
    name: home.docSet,
    value: 0,
  },
  {
    name: home.qaSet,
    value: 1,
  },
];

const columns = [
  {
    title: home.documentId,
    dataIndex: 'id',
    key: 'id',
    width: '11%',
  },
  {
    title: home.documentName,
    dataIndex: 'fileIdName',
    key: 'fileIdName',
    width: '20%',
    ellipsis: true,
  },
  {
    title: home.documentStatus,
    dataIndex: 'status',
    key: 'status',
    width: '15%',
    ellipsis: true,
  },
  {
    title: home.fileSize,
    dataIndex: 'bytes',
    key: 'bytes',
    width: '10%',
  },
  {
    title: home.creationDate,
    dataIndex: 'createtime',
    key: 'createtime',
    width: '11%',
  },
  {
    title: home.remark,
    dataIndex: 'errortext',
    key: 'errortext',
    width: '15%',
  },
  {
    title: home.operate,
    key: 'options',
    width: '10%',
  },
];

const qaColumns = [
  {
    title: 'ID',
    dataIndex: 'id',
    key: 'id',
    width: '8%',
  },
  {
    title: home.question,
    dataIndex: 'question',
    key: 'question',
    width: '43%',
    ellipsis: true,
  },
  {
    title: home.status,
    dataIndex: 'status',
    key: 'status',
    width: '10%',
    ellipsis: true,
  },
  {
    title: home.characterCount,
    dataIndex: 'bytes',
    key: 'bytes',
    width: '10%',
  },
  {
    title: home.creationDate,
    dataIndex: 'createtime',
    key: 'createtime',
    width: '11%',
  },
  {
    title: home.operate,
    key: 'options',
    width: '10%',
  },
];

const paginationConfig = computed(() => ({
  current: pageNum.value, // 当前页码
  pageSize: 10, // 每页条数
  total: total.value, // 数据总数
  showSizeChanger: false,
  showTotal: total => `共 ${total} 条`,
}));

let optionItem: any = {};

const deleteItem = item => {
  optionItem = item;
};

const confirm = async () => {
  try {
    await resultControl(
      await urlResquest.deleteFile({ fileIds: [optionItem.fileId], kbId: currentId.value })
    );
    message.success('删除成功');
    getDetails();
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
};

let qaOptionItem: any = {};

const deleteQaItem = item => {
  qaOptionItem = item;
};

const qaConfirm = async () => {
  console.log(qaOptionItem);
  try {
    await resultControl(
      await urlResquest.deleteFile({
        kb_id: `${currentId.value}_FAQ`,
        file_ids: [qaOptionItem.faqId],
      })
    );
    message.success('删除成功');
    getFaqList();
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
};

const editQaItem = item => {
  setFaqType('edit');
  setEditQaSet(item);
  setEditModalVisible(true);
};
const goChat = () => {
  setDefault(pageStatus.normal);
};

const showFileUpload = () => {
  setModalVisible(true);
  setModalTitle(home.upload);
};

const showUrlUpload = () => {
  setUrlModalVisible(true);
  setModalTitle(common.addUrl);
};

const showEditQaSet = () => {
  setFaqType('upload');
  setEditModalVisible(true);
  console.log('showEditQaSet');
};

const parseStatus = status => {
  console.log('status', status);
  let str = common.failed;
  switch (status) {
    case 'gray':
      str = common.parsing;
      break;
    case 'green':
      str = common.succeeded;
      break;
    default:
      break;
  }
  return str;
};

const parseFaqStatus = status => {
  let str = common.failed;
  switch (status) {
    case 'gray':
      str = common.uploadCompleted;
      break;
    case 'green':
      str = common.learningCompleted;
      break;
    default:
      break;
  }
  return str;
};

const checkKbIsCreate = async () => {
  try {
    const res: any = await resultControl(await urlResquest.kbList());
    if (res.find(item => item.kb_id === `${currentId.value}_FAQ`)) {
      return true;
    }
  } catch (e) {
    message.error(e.msg || common.error);
    return true;
  }
  return false;
};

const addKnowledge = async () => {
  const isCreate = await checkKbIsCreate();
  if (isCreate) {
    return;
  }
  //获取到知识库id后  赋值给newId
  try {
    const res: any = await resultControl(
      await urlResquest.createKb({
        kb_name: `${currentKbName.value}_FAQ`,
        kb_id: `${currentId.value}_FAQ`,
      })
    );
    console.log(res);
  } catch (e) {
    message.error(e.msg || common.error);
    return false;
  }
  return true;
};

const navClick = value => {
  navIndex.value = value;
  if (value === 0) {
    clearTimeout(faqTimer.value);
    getDetails();
  } else {
    setPageNum(1);
    clearTimeout(timer.value);
    getFaqList();
    addKnowledge();
  }
};

const onChange = pagination => {
  console.log('onChange', pagination, paginationConfig);
  const { current } = pagination;
  setPageNum(current);
};

watch(
  currentId,
  () => {
    console.log('current id changed');
    navIndex.value = 0;
    setPageNum(1);
    getDetails();
  },
  {
    immediate: true,
  }
);

onBeforeUnmount(() => {
  clearTimeout(timer.value);
  clearTimeout(faqTimer.value);
  console.log('销毁请求');
});
</script>

<style lang="scss" scoped>
.list-page {
  overflow: hidden;
  width: 100%;
  height: 100%;
  background-color: $baseColor;
  font-family: PingFang SC;

  .content {
    height: calc(100vh - 64px);
    // margin-top: 16px;
    padding: 24px 32px;
    background: #f3f6fd;
    border-radius: 12px 0 0 0;
  }
}

.options {
  display: flex;
  align-items: center;
  .to-chat {
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 40px;
    background: #5a47e5;
    border-radius: 6px;
    padding: 8px 20px;
    img {
      margin-right: 4px;
      width: 20px;
      height: 20px;
    }
    span {
      font-size: 16px;
      font-weight: 500;
      line-height: 24px;
      color: #ffffff;
    }
  }

  .kb-name {
    margin-left: 20px;
    margin-right: auto;
    font-size: 24px;
    font-weight: 500;
    line-height: 32px;
    color: #222222;

    .name {
      margin-right: 20px;
    }

    .id {
      font-size: 12px;
    }
  }
}

.nav-info {
  width: 100%;
  height: 40px;
  margin: 20px 0 14px 0;
  display: flex;
  justify-content: space-between;
  .navs {
    height: 40px;
    padding: 4px;
    border-radius: 8px;
    background: #e4e9f4;
    display: flex;
    .nav-item {
      // width: 100px;
      padding: 0 24px;
      height: 32px;
      font-size: 16px;
      color: #666666;
      border-radius: 6px;
      text-align: center;
      line-height: 32px;
      cursor: pointer;
    }
    .nav-item-active {
      background: #fff;
      font-weight: 500;
      color: #5a47e5;
    }
  }
  .handle-btn {
    display: flex;
    .upload {
      cursor: pointer;
      height: 40px;
      padding: 8px 20px;
      border-radius: 4px;
      background: #5a47e5;
      font-size: 16px;
      font-weight: 500;
      line-height: 24px;
      color: #ffffff;
    }

    .add-link {
      cursor: pointer;
      margin-left: 16px;
      padding: 8px 20px;
      border-radius: 4px;
      background: #ffffff;
      border: 1px solid #5a47e5;
      font-size: 16px;
      font-weight: 500;
      line-height: 24px;
      color: #5a47e5;
    }
  }
}

.table {
  // margin-top: 32px;
  margin-bottom: 32px;
  height: calc(100% - 120px);
  overflow: auto;
  border-radius: 12px;
  background-color: #fff;

  .options {
    width: 80px;
    display: flex;
    justify-content: space-between;
  }

  .delete-item {
    padding: 0;
    font-size: 14px;
    font-weight: normal;
    line-height: 22px;
    /* 错误颜色 */
    color: #ff524c;
    cursor: pointer;
  }

  .edit-item {
    padding: 0;
  }

  .file-status {
    width: 16px;
    height: 16px;
  }

  .status-box {
    display: flex;
    align-items: center;
    .icon-file-status {
      display: flex;
      align-items: center;
    }
    span {
      display: block;

      margin-right: 8px;
      svg {
        width: 16px;
        height: 16px;
      }
    }
  }
}

:deep(.ant-table-wrapper .ant-table-thead > tr > th) {
  font-size: 16px !important;
  font-weight: 500 !important;
  line-height: 24px !important;
  padding: 15px 0 15px 36px !important;

  color: #222222 !important;
  background-color: #e9edf7;
  .small {
    font-size: 12px !important;
  }

  &:before {
    width: 0 !important;
  }
}

:deep(.ant-table-tbody > tr > td) {
  font-size: 14px;
  font-weight: normal;
  line-height: 22px;
  color: #666666;
  background-color: #fff;
  padding: 20px 0 20px 36px !important;
  border: 0 !important;
  box-shadow: inset 0px -1px 0px 0px rgba(0, 0, 0, 0.05);

  &:hover {
    background-color: rgba(233, 237, 247, 0.3);
  }
}

// :deep(.ant-pagination-item-link) {
//   border-radius: 4px !important;
//   box-sizing: border-box !important;
//   border: 1px solid #dde2ec !important;
// }

:deep(.ant-pagination) {
  margin: 16px 20px !important;
}

:deep(.ant-pagination-item) {
  box-sizing: border-box !important;
  border: 1px solid #dde2ec !important;
}

:deep(.ant-pagination-item-active) {
  background: #5a47e5 !important;
  color: #fff !important;
  a {
    color: #fff !important;
  }
}

:deep(.options > .ant-btn) {
  height: auto;
}

:deep(.ant-btn-link) {
  color: #5a47e5;
}

:deep(.ant-btn-link:disabled) {
  color: rgba(0, 0, 0, 0.25);
}

:deep(.ant-table-empty .ant-table-placeholder .ant-table-cell) {
  color: #999999 !important;
}
</style>

<style lang="scss">
.del-pop {
  margin-right: 10px;
  .ant-popover-content {
    .ant-btn-default {
      padding: 1px 8px;
      border: 1px solid rgba(0, 0, 0, 0.15) !important;

      span {
        line-height: 1;
      }
    }
    .ant-btn-primary {
      background-color: rgba(90, 71, 229, 1) !important;
      color: #ffffff;
      padding: 1px 8px;
    }
    .ant-popover-inner {
      padding: 12px 16px;
      transform: translateX(44px);
    }

    .ant-popconfirm-message-icon {
      svg {
        font-size: 16px;
      }
    }
    .ant-popconfirm-message-title {
      width: 168px;
      height: 36px;
      line-height: 36px;
    }

    .ant-popconfirm-message {
      align-items: center !important;
    }
  }
}
.qa-del-pop {
  .ant-popover-inner {
    padding-top: 20px;
    height: 100px;
  }
  .ant-popconfirm-buttons {
    margin-top: 16px;
  }
  .ant-btn-primary {
    background-color: rgba(90, 71, 229, 1) !important;
    color: #ffffff;
    padding: 1px 8px;
  }
  .ant-btn-sm {
    width: 60px;
  }
  .ant-btn-sm.ant-btn-loading {
    width: auto !important;
  }
}
</style>
