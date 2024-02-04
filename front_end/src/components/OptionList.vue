<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-26 14:49:41
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-05 19:08:03
 * @FilePath: /qanything-open-source/src/components/OptionList.vue
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
        <p class="kb-name">{{ currentKbName }}</p>
        <div class="upload" @click="showFileUpload">{{ home.upload }}</div>
        <div class="add-link" @click="showUrlUpload">{{ home.addUrl }}</div>
      </div>
      <div class="table">
        <a-table
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
const { getDetails } = useOptiionList();
const { dataSource, timer } = storeToRefs(useOptiionList());
import { getLanguage } from '@/language/index';

const common = getLanguage().common;
const home = getLanguage().home;

const columns = [
  {
    title: home.documentId,
    dataIndex: 'id',
    key: 'id',
    width: '11%',
  },
  {
    title: home.documentName,
    dataIndex: 'file_name',
    key: 'file_name',
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
    width: '10%',
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

let optionItem: any = {};

const deleteItem = item => {
  optionItem = item;
};

const confirm = async () => {
  try {
    await resultControl(
      await urlResquest.deleteFile({ file_ids: [optionItem.file_id], kb_id: currentId.value })
    );
    message.success('删除成功');
    getDetails();
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
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

const parseStatus = status => {
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

watch(
  currentId,
  () => {
    console.log('current id changed');
    getDetails();
  },
  {
    immediate: true,
  }
);

onBeforeUnmount(() => {
  clearTimeout(timer.value);
  console.log('销毁请求');
});
</script>

<style lang="scss" scoped>
.list-page {
  overflow: hidden;
  width: 100%;
  height: 100%;
  background-color: $baseColor;

  .content {
    height: calc(100vh - 64px - 16px);
    margin-top: 16px;
    padding: 32px;
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
  }

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

.table {
  margin-top: 32px;
  margin-bottom: 32px;
  height: calc(100% - 90px);
  overflow: auto;
  border-radius: 12px;
  background-color: #fff;

  .delete-item {
    font-size: 14px;
    font-weight: normal;
    line-height: 22px;
    /* 错误颜色 */
    color: #ff524c;
    cursor: pointer;
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
  padding: 20px 0 20px 36px !important;

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
  padding: 40px 0 40px 36px !important;
  border: 0 !important;
  box-shadow: inset 0px -1px 0px 0px rgba(0, 0, 0, 0.05);

  &:hover {
    background-color: rgba(233, 237, 247, 0.3);
  }
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
</style>
