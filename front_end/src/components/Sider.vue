<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-31 15:40:55
 * @FilePath: front_end/src/components/Sider.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div class="sider">
    <div v-if="navIndex === 0" class="knowledge">
      <div class="add-btn">
        <!-- <AddInput @add="addKb" /> -->
        <AddInput />
      </div>
      <div class="content">
        <SiderCard :list="knowledgeBaseList"></SiderCard>
      </div>
      <!-- <div class="bottom-btn-box">
        <a-button class="manage" @click="goManage">
          <template #icon>
            <img class="folder" src="../assets/home/icon-folder.png" alt="图标" />
          </template>
          知识库管理</a-button
        >
      </div> -->
    </div>
    <div v-else-if="navIndex === 1" class="bots">
      <div class="bots-tab" @click="changePage('/bots')">{{ getLanguage().bots.myBots }}</div>
      <NewBotsDialog />
      <SelectKnowledgeDialog />
      <CopyUrlDialog />
    </div>
    <div v-else-if="navIndex === 2" class="quick-start">
      <div class="content">
        <div
          :class="['card-new', chatId === null ? 'active' : '', showLoading ? 'disabled' : '']"
          @click="quickClickHandle(0)"
        >
          <SvgIcon name="new-chat" />
          {{ getLanguage().home.newConversation }}
        </div>
        <SiderCardItem
          v-for="item of historyList"
          :key="item.historyId"
          :card-data="item"
          @click="quickClickHandle(1, item)"
        />
      </div>
    </div>
    <ChatSourceDialog />
    <DeleteModal />
    <UrlUploadDialog />
    <EditQaSetDialog />
  </div>
</template>
<script lang="ts" setup>
// import dayjs from 'dayjs';
import AddInput from '@/components/AddInput.vue';
import SiderCard from '@/components/SiderCard.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
// import FileUploadDialog from '@/components/FileUploadDialog.vue';
import UrlUploadDialog from '@/components/UrlUploadDialog.vue';
import DeleteModal from '@/components/DeleteModal.vue';
import EditQaSetDialog from '@/components/EditQaSetDialog.vue';
import NewBotsDialog from '@/components/Bots/NewBotsDialog.vue';
import SelectKnowledgeDialog from '@/components/Bots/SelectKnowledgeDialog.vue';
import CopyUrlDialog from '@/components/Bots/CopyUrlDialog.vue';
import ChatSourceDialog from '@/components/ChatSourceDialog.vue';
import { useHeader } from '@/store/useHeader';
import routeController from '@/controller/router';
import SiderCardItem from '@/components/SiderCardItem.vue';
import { IHistoryList, useQuickStart } from '@/store/useQuickStart';
import SvgIcon from '@/components/SvgIcon.vue';
import { getLanguage } from '@/language';
// import { message } from 'ant-design-vue';
// import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { message } from 'ant-design-vue';
// import urlResquest from '@/services/urlConfig';
// import { pageStatus } from '@/utils/enum';
// import { resultControl } from '@/utils/utils';

// const { setModalVisible } = useKnowledgeModal();
// const { modalVisible } = storeToRefs(useKnowledgeModal());

// const router = useRouter();
// const { getList, setCurrentId, setCurrentKbName, setDefault } = useKnowledgeBase();
// const { knowledgeBaseList, selectList } = storeToRefs(useKnowledgeBase());
const { knowledgeBaseList } = storeToRefs(useKnowledgeBase());
const { historyList, showLoading, chatId, QA_List, kbId } = storeToRefs(useQuickStart());
const { getChatById } = useQuickStart();
const { navIndex } = storeToRefs(useHeader());
const { changePage } = routeController();

// 快速开始逻辑
function addQuestion(q) {
  QA_List.value.push({
    question: q,
    type: 'user',
  });
  // scrollBottom();
}

function addAnswer(question: string, answer: string, picList, qaId, source) {
  QA_List.value.push({
    answer,
    question,
    type: 'ai',
    qaId,
    copied: false,
    like: false,
    unlike: false,
    source: source ? source : [],
    showTools: true,
    picList,
  });
}

// 0新建对话, 1选中对话
const quickClickHandle = (type: 0 | 1, cardData?: IHistoryList) => {
  if (showLoading.value) return;
  if (type === 0) {
    if (chatId.value === null) {
      message.info('已切换最新对话');
      return;
    }
    chatId.value = null;
    QA_List.value = [];
    kbId.value = '';
  } else if (type === 1) {
    if (chatId.value === cardData.historyId) return;
    chatId.value = cardData.historyId;
    QA_List.value = [];
    kbId.value = cardData.kbId;
    // try {
    const chat = getChatById(chatId.value);
    chat.list.forEach(item => {
      if (item.type === 'user') {
        addQuestion(item.question);
      } else if (item.type === 'ai') {
        addAnswer(item.question, item.answer, item.picList, item.qaId, item.source);
      }
    });
    // } catch (e) {
    //   message.error(e.msg || '获取问答历史失败');
    // }
  }
};

onUnmounted(() => {
  QA_List.value = [];
  chatId.value = null;
  kbId.value = '';
});

//创建知识库
// const addKb = async kbName => {
//   console.log(kbName);
//   if (!kbName.length) {
//     message.error('请输入知识库名称');
//     return;
//   }

//   try {
//     const res: any = await resultControl(await urlResquest.createKb({ kbName: kbName }));

//     console.log(res);
//     setCurrentId(res?.kbId);
//     setCurrentKbName(res?.kbName);
//     selectList.value.push(res?.kbId);
//     await getList();
//     setModalVisible(!modalVisible.value);
//     setDefault(pageStatus.optionlist);
//   } catch (e) {
//     console.log(e);
//     message.error(e.msg || '请求失败');
//   }
// };
</script>

<style lang="scss" scoped>
.sider {
  display: flex;
  flex-direction: column;
  width: 280px;
  height: calc(100vh - 64px);
  background-color: #26293b;

  .knowledge {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .quick-start {
    height: 100%;
    display: flex;
    flex-direction: column;

    .card-new {
      width: 232px;
      height: 48px;
      margin: 0 auto 16px;
      border-radius: 8px;
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: 16px;
      overflow: hidden;
      background: #333647;
      cursor: pointer;
      color: #fff;

      svg {
        width: 16px;
        height: 16px;
        margin-right: 4px;
        margin-top: 2px;
      }
    }

    .active {
      background: linear-gradient(284deg, #7b5ef2 -1%, #c383fe 97%);
    }

    .disabled {
      cursor: not-allowed !important;

      span {
        cursor: not-allowed !important;
      }

      .close-icon {
        cursor: not-allowed !important;
      }

      .close-icon:hover {
        background: transparent !important;
      }
    }
  }

  .add-btn {
    margin: 14px 24px 20px 24px;
    width: calc(100% - 48px);

    :deep(.ant-input-affix-wrapper) {
      padding: 4px;
      border: 1px solid #373b4d;
      background: linear-gradient(0deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), #26293b;
    }

    :deep(.ant-input) {
      color: #ffffff;
      padding-left: 4px;
      background: linear-gradient(0deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), #26293b;

      &::placeholder {
        color: #999999;
      }
    }
  }

  .bottom-btn-box {
    position: fixed;
    width: 280px;
    bottom: 29px;

    .manage {
      width: calc(100% - 40px);
      margin: 0 20px;
      height: 40px;
    }

    .folder {
      width: 16px;
      height: 16px;
      margin-right: 8px;
    }

    :deep(.ant-btn) {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #4d71ff !important;
      background: rgba(255, 255, 255, 0.7) !important;
      border: 1px solid #ffffff !important;
    }
  }

  .bots {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 12px;

    .bots-tab {
      width: 232px;
      height: 46px;
      border-radius: 8px;
      background: #7261e9;
      font-family: PingFang SC;
      font-size: 16px;
      font-weight: 500;
      text-align: center;
      line-height: 46px;
      color: #fff;
      cursor: pointer;
    }
  }
}

.content {
  flex: 1;
  margin-bottom: 20px;
  margin-top: 20px;
  overflow-y: scroll;
}
</style>
