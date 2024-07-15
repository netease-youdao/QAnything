<template>
  <div class="history-chat">
    <div class="history-chat-content">
      <div class="list">
        <div
          class="new-chat"
          :class="[showLoading ? 'disabled' : '', chatId === null ? 'new-active' : '']"
          @click="newChat"
        >
          <SvgIcon name="new-chat" />
          {{ getLanguage().home.newConversation }}
        </div>
        <div
          v-for="(item, index) in chatList"
          :key="item.historyId"
          :class="[
            'chat-item',
            chatId === item.historyId ? 'item-active' : '',
            showLoading ? 'disabled' : '',
          ]"
        >
          <div class="line"></div>
          <span @click="changeChat(item)">{{ item.title }}</span>
          <img
            v-if="chatId === item.historyId"
            src="@/assets/home/close-icon.png"
            class="close-icon"
            alt="close"
            @click="deleteChat(item, index)"
          />
        </div>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { useHomeChat } from '@/store/useHomeChat';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import SvgIcon from '@/components/SvgIcon.vue';
import { getLanguage } from '@/language';

const props = defineProps({
  observer: {
    type: Object,
    required: true,
  },
  observeDom: {
    type: Object,
    default: null,
  },
  qaObserver: {
    type: Object,
    required: true,
  },
  qaObserveDom: {
    type: Object,
    default: null,
  },
  showLoading: {
    type: Boolean,
    require: true,
  },
});

const { chatList, chatId, QA_List, qaPageId, pageId } = storeToRefs(useHomeChat());
const { setSelectList } = useKnowledgeBase();

const emits = defineEmits(['scrollBottom', 'setObserveDom', 'setQaObserverDom', 'clearHistory']);

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

// 选择/切换对话
async function changeChat(item) {
  // 正在问答时禁止操作
  if (props.showLoading) {
    return;
  }
  chatId.value = item.historyId;
  QA_List.value = [];
  qaPageId.value = 1;
  setSelectList([...item.kbIds]);
  emits('clearHistory');
  try {
    const res: any = await resultControl(
      await urlResquest.chatDetail({
        historyId: chatId.value,
        page: qaPageId.value,
        pageSize: 50,
      })
    );
    // 清除上次监听的dom元素
    if (props.qaObserveDom !== null) {
      props.qaObserver.unobserve(props.qaObserveDom);
      emits('setQaObserverDom', null);
    }
    res.detail.reverse().forEach(item => {
      addQuestion(item.question);
      addAnswer(item.question, item.answer, item.picList, item.qaId, item.source);
    });
    emits('scrollBottom');
    if (res.detail.length >= 50) {
      await nextTick(() => {
        // 监听新的dom元素
        const eles: any = document.getElementsByClassName('chat-li');
        if (eles.length) {
          props.qaObserver.observe(eles[0]);
          emits('setQaObserverDom', eles[0]);
        }
      });
    }
  } catch (e) {
    message.error(e.msg || '获取问答历史失败');
  }
}

function newChat() {
  if (props.showLoading) {
    return;
  }
  if (chatId.value === null) {
    message.info('已切换最新对话');
    return;
  }
  chatId.value = null;
  QA_List.value = [];
  qaPageId.value = 1;
  pageId.value = 1;
  emits('clearHistory');
}

async function deleteChat(item, index) {
  if (props.showLoading) {
    return;
  }
  try {
    await resultControl(await urlResquest.deleteChat({ historyId: item.historyId }));
    if (item.historyId === chatId.value) {
      newChat();
    }
    message.success('删除成功');
    chatList.value.splice(index, 1);
    await nextTick(() => {
      resetObserve();
    });
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
}

function resetObserve() {
  if (props.qaObserveDom !== null) {
    props.qaObserver.unobserve(props.qaObserveDom);
    emits('setQaObserverDom', null);
    const eles: any = document.getElementsByClassName('chat-li');
    if (eles.length) {
      props.qaObserver.observe(eles[0]);
      emits('setQaObserverDom', eles[0]);
    }
  }
}
</script>
<style lang="scss" scoped>
.history-chat {
  height: 64px;
  background: #f3f6fd;
  border-bottom: 1px solid #dedede;
  padding: 0 40px;
  border-radius: 12px 0 0 0;
  display: flex;
  align-items: center;

  &::-webkit-scrollbar {
    display: none;
  }

  .close {
    margin-right: 16px;
    margin-top: 4px;

    img {
      cursor: pointer;
      width: 16px;
      height: 16px;
    }
  }

  .history-chat-content {
    width: 100%;
    overflow-x: auto;

    &::-webkit-scrollbar {
      display: none;
    }
  }

  .list {
    display: flex;

    .new-chat {
      flex-shrink: 0;
      font-size: 16px;
      font-weight: normal;
      line-height: 24px;
      color: #666666;
      cursor: pointer;
      display: flex;
      align-items: center;

      svg {
        width: 16px;
        height: 16px;
        margin-right: 4px;
        margin-top: 2px;
      }
    }

    .new-active {
      color: #5a47e5;
    }

    .chat-item {
      max-width: 160px;
      // margin-left: 56px;
      display: flex;
      align-items: center;

      .line {
        width: 1px;
        height: 16px;
        background: #d8d8d8;
        margin: 0 20px;
      }

      span {
        font-size: 16px;
        font-weight: normal;
        line-height: 24px;
        color: #666666;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        cursor: pointer;
      }

      img {
        width: 16px;
        height: 16px;
        margin-left: 4px;
        margin-top: 2px;
        cursor: pointer;
      }

      .close-icon:hover {
        background: #f0f0f0;
        border-radius: 3px;
      }
    }

    .item-active {
      span {
        color: #5a47e5;
      }
    }
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
</style>
