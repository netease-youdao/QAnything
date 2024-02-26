<template>
  <div class="container">
    <div class="my-page">
      <div id="chat" class="chat">
        <ul id="chat-ul" ref="scrollDom">
          <li v-for="(item, index) in QA_List" :key="index">
            <div v-if="item.type === 'user'" class="user">
              <img class="avatar" src="../assets/home/avatar.png" alt="头像" />
              <p class="question-text">{{ item.question }}</p>
            </div>
            <div v-else class="ai">
              <div class="content">
                <img class="avatar" src="../assets/home/ai-avatar.png" alt="头像" />
                <p
                  class="question-text"
                  :class="[
                    !item.source.length ? 'change-radius' : '',
                    item.showTools ? '' : 'flashing',
                  ]"
                  v-html="item.answer"
                ></p>
              </div>
              <template v-if="item.source.length">
                <div
                  v-for="(sourceItem, sourceIndex) in item.source"
                  :key="sourceIndex"
                  class="data-source"
                >
                  <p v-show="sourceItem.file_name" class="control">
                    <span class="tips">{{ common.dataSource }}{{ sourceIndex + 1 }}:</span
                    ><span class="file">{{ sourceItem.file_name }}</span>
                    <SvgIcon
                      v-show="sourceItem.showDetailDataSource"
                      name="iconup"
                      @click="showDetail(item, sourceIndex)"
                    />
                    <SvgIcon
                      v-show="!sourceItem.showDetailDataSource"
                      name="icondown"
                      @click="showDetail(item, sourceIndex)"
                    />
                  </p>
                  <Transition name="sourceitem">
                    <div class="source-content">
                      <p
                        v-show="sourceItem.showDetailDataSource"
                        v-html="sourceItem.content.replaceAll('\n', '<br/>')"
                      ></p>
                      <p class="score">
                        <span class="tips">{{ common.correlation }}</span
                        >{{ sourceItem.score }}
                      </p>
                    </div>
                  </Transition>
                </div>
              </template>
              <div v-if="item.showTools" class="feed-back">
                <div class="reload-box" @click="reAnswer(item)">
                  <SvgIcon name="reload"></SvgIcon>
                  <span class="reload-text">{{ common.regenerate }}</span>
                </div>
                <div class="tools">
                  <SvgIcon
                    :style="{
                      color: item.copied ? '#4D71FF' : '',
                    }"
                    name="copy"
                    @click="myCopy(item)"
                  ></SvgIcon>
                  <SvgIcon
                    :style="{
                      color: item.like ? '#4D71FF' : '',
                    }"
                    name="like"
                    @click="like(item, $event)"
                  ></SvgIcon>
                  <SvgIcon
                    :style="{
                      color: item.unlike ? '#4D71FF' : '',
                    }"
                    name="unlike"
                    @click="unlike(item)"
                  ></SvgIcon>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div class="stop-btn">
        <a-button v-show="showLoading" @click="stopChat">
          <template #icon>
            <SvgIcon name="stop" :class="showLoading ? 'loading' : ''"></SvgIcon> </template
          >{{ common.stop }}</a-button
        >
      </div>
      <div class="question-box">
        <div class="question">
          <a-popover placement="topLeft">
            <template #content>
              <p v-if="control">退出多轮对话</p>
              <p v-else>开启多轮对话</p>
            </template>
            <span :class="['control', `control-${control}`]">
              <SvgIcon name="chat-control" @click="controlChat" />
            </span>
          </a-popover>
          <span class="download" @click="downloadChat">
            <SvgIcon name="chat-download" />
          </span>
          <span class="delete" @click="deleteChat">
            <SvgIcon name="chat-delete" />
          </span>
          <a-input
            v-model:value="question"
            max-length="200"
            :placeholder="common.problemPlaceholder"
            @keyup.enter="send"
          >
            <template #suffix>
              <div class="send-plane">
                <a-button type="primary" :disabled="showLoading" @click="send">
                  <SvgIcon name="sendplane"></SvgIcon>
                </a-button>
              </div>
            </template>
          </a-input>
        </div>
      </div>
    </div>
  </div>
  <DefaultModal :content="content" :confirm-loading="confirmLoading" @ok="confirm" />
</template>
<script lang="ts" setup>
import { apiBase } from '@/services';
import { IChatItem } from '@/utils/types';
import { useThrottleFn, useClipboard } from '@vueuse/core';
import { message } from 'ant-design-vue';
import SvgIcon from './SvgIcon.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useChat } from '@/store/useChat';
import { Typewriter } from '@/utils/typewriter';
import DefaultModal from './DefaultModal.vue';
import html2canvas from 'html2canvas';
import { userId } from '@/services/urlConfig';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;

const typewriter = new Typewriter((str: string) => {
  if (str) {
    QA_List.value[QA_List.value.length - 1].answer += str || '';
  }
});

const { selectList } = storeToRefs(useKnowledgeBase());
const { QA_List } = storeToRefs(useChat());
const { copy } = useClipboard();
declare module _czc {
  const push: (array: any) => void;
}

//当前是否多轮对话
const control = ref(true);

//当前问的问题
const question = ref('');

//问答的上下文
const history = ref([]);

//当前是否回答中
const showLoading = ref(false);

//取消请求用
let ctrl: AbortController;

const scrollDom = ref(null);

const scrollBottom = () => {
  nextTick(() => {
    scrollDom.value?.scrollIntoView(false);
  });
};

const like = useThrottleFn((item, e) => {
  item.like = !item.like;
  item.unlike = false;
  _czc.push(['_trackEvent', 'qanything', '问答页面', '点赞', '', '']);
  if (item.like) {
    e.target.parentNode.style.animation = 'shake ease-in .5s';
    const timer = setTimeout(() => {
      clearTimeout(timer);
      e.target.parentNode.style.animation = '';
    }, 600);
  }
}, 800);
const unlike = (item: IChatItem) => {
  item.unlike = !item.unlike;
  item.like = false;
  _czc.push(['_trackEvent', 'qanything', '问答页面', '点踩', '', '']);
};

//拷贝
const myCopy = (item: IChatItem) => {
  copy(item.answer)
    .then(() => {
      item.copied = !item.copied;
      message.success(common.copySuccess, 1);
      const timer = setTimeout(() => {
        clearTimeout(timer);
        item.copied = !item.copied;
      }, 1000);
    })
    .catch(() => {
      message.error(common.copyFailed, 1);
    });
};

const addQuestion = q => {
  QA_List.value.push({
    question: q,
    type: 'user',
  });
  scrollBottom();
};

const addAnswer = (question: string) => {
  QA_List.value.push({
    answer: '',
    question,
    type: 'ai',
    copied: false,
    like: false,
    unlike: false,
    source: [],
    showTools: false,
  });
};

const stopChat = () => {
  if (ctrl) {
    ctrl.abort();
  }
  typewriter.done();
  showLoading.value = false;
  QA_List.value[QA_List.value.length - 1].showTools = true;
};

//发送问答消息
const send = () => {
  if (!question.value.length) {
    return;
  }
  if (!selectList.value.length) {
    return message.warning(common.chooseError);
  }
  const q = question.value;
  question.value = '';
  addQuestion(q);
  if (history.value.length >= 3) {
    history.value = [];
  }
  showLoading.value = true;
  ctrl = new AbortController();

  fetchEventSource(apiBase + '/local_doc_qa/local_doc_chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: ['text/event-stream', 'application/json'],
    },
    body: JSON.stringify({
      user_id: userId,
      kb_ids: selectList.value,
      history: control.value ? history.value : [],
      question: q,
      streaming: true,
    }),
    signal: ctrl.signal,
    onopen(e: any) {
      console.log('open');
      if (e.ok && e.headers.get('content-type') === 'text/event-stream') {
        console.log("everything's good");
        // addAnswer(question.value);
        // question.value = '';
        addAnswer(q);
        typewriter.start();
      } else if (e.headers.get('content-type') === 'application/json') {
        showLoading.value = false;
        return e
          .json()
          .then(data => {
            message.error(data?.msg || '出错了,请稍后刷新重试。');
          })
          .catch(e => {
            console.log(e);
            message.error('出错了,请稍后刷新重试。');
          }); // 将响应解析为 JSON
      }
    },
    onmessage(msg: { data: string }) {
      console.log('message');
      const res: any = JSON.parse(msg.data);
      console.log(res);
      if (res?.code == 200 && res?.response) {
        // QA_List.value[QA_List.value.length - 1].answer += res.result.response;
        typewriter.add(res?.response.replaceAll('\n', '<br/>'));
        scrollBottom();
      }

      if (res?.source_documents?.length) {
        QA_List.value[QA_List.value.length - 1].source = res?.source_documents;
      }

      if (res?.history.length) {
        history.value = res?.history;
      }
    },
    onclose(e: any) {
      console.log('close');
      console.log(e);
      typewriter.done();
      ctrl.abort();
      showLoading.value = false;
      QA_List.value[QA_List.value.length - 1].showTools = true;
      nextTick(() => {
        scrollBottom();
      });
    },
    onerror(err: any) {
      console.log('error');
      typewriter?.done();
      ctrl?.abort();
      showLoading.value = false;
      QA_List.value[QA_List.value.length - 1].showTools = true;
      message.error(err.msg || '出错了');
      nextTick(() => {
        scrollBottom();
      });
      throw err;
    },
  });
};

const reAnswer = (item: IChatItem) => {
  console.log('reAnswer');
  question.value = item.question;
  send();
};

//点击查看是否显示详细来源
const showDetail = (item: IChatItem, index) => {
  item.source[index].showDetailDataSource = !item.source[index].showDetailDataSource;
};

//下载 清除聊天记录相关
const { showModal } = storeToRefs(useChat());
const { clearQAList } = useChat();
const confirmLoading = ref(false);
const content = ref('');
const type = ref('');
const downloadChat = () => {
  type.value = 'download';
  showModal.value = true;
  content.value = common.saveTip;
};

const controlChat = () => {
  control.value = !control.value;
};

const deleteChat = () => {
  type.value = 'delete';
  showModal.value = true;
  content.value = common.clearTip;
};

const confirm = async () => {
  confirmLoading.value = true;
  if (type.value === 'download') {
    console.log('download');
    try {
      const ele = document.getElementById('chat-ul');
      const canvas = await html2canvas(ele as HTMLDivElement, {
        useCORS: true,
      });
      const imgUrl = canvas.toDataURL('image/png');
      const tempLink = document.createElement('a');
      tempLink.style.display = 'none';
      tempLink.href = imgUrl;
      tempLink.setAttribute('download', 'chat-shot.png');
      if (typeof tempLink.download === 'undefined') tempLink.setAttribute('target', '_blank');

      document.body.appendChild(tempLink);
      tempLink.click();
      document.body.removeChild(tempLink);
      window.URL.revokeObjectURL(imgUrl);
      message.success('下载成功');
      Promise.resolve();
    } catch (e) {
      console.log(e);
      message.error(e.message || e.msg || '出错了');
    }
  } else if (type.value === 'delete') {
    console.log('delete');
    history.value = [];
    clearQAList();
  }
  type.value = '';
  content.value = '';
  confirmLoading.value = false;
  showModal.value = false;
};
scrollBottom();
</script>

<style lang="scss" scoped>
.container {
  padding-top: 16px;
  background-color: #26293b;
}
.my-page {
  position: relative;
  margin: 0 auto;
  border-radius: 12px 0 0 0;
  background: #f3f6fd;
}
.chat {
  margin: 0 auto;
  width: 75.36%;
  min-width: 900px;
  max-width: 1239px;
  height: calc(100vh - 54px - 48px - 28px - 28px - 32px - 50px);
  overflow-y: auto;
  padding-top: 28px;

  #chat-ul {
    background: #f3f6fd;
  }

  .avatar {
    width: 32px;
    height: 32px;
    margin-right: 16px;
  }

  .user {
    display: flex;
    margin-bottom: 16px;

    .question-text {
      padding: 13px 20px;
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      color: #222222;
      background: #e9e1ff;
      border-radius: 0px 12px 12px 12px;
      word-wrap: break-word;
    }
  }

  .ai {
    margin: 16px 0 28px 0;
    .content {
      display: flex;

      .question-text {
        flex: 1;
        padding: 13px 20px;
        font-size: 14px;
        font-weight: normal;
        line-height: 22px;
        color: $title1;
        background: #fff;
        border-radius: 0px 12px 0px 0px;
        word-wrap: break-word;
      }

      .flashing {
        &:after {
          -webkit-animation: blink 1s steps(5, start) infinite;
          animation: blink 1s steps(5, start) infinite;
          content: '▋';
          margin-left: 0.25rem;
          vertical-align: baseline;
        }
      }

      .change-radius {
        border-radius: 0px 12px 12px 12px;
      }
    }

    .data-source {
      margin-left: 48px;
      padding: 13px 20px;
      font-size: 14px;
      line-height: 22px;
      color: $title1;
      background: #fff;

      &:nth-last-of-type(2) {
        border-radius: 0px 0px 12px 12px;
      }

      &:nth-first-of-type(1) {
        border-radius: 0px 12px 12px 12px;
      }
      .control {
        display: flex;
        align-items: center;
      }

      .score {
        margin-top: 26px;
      }

      .source-content {
        margin-top: 26px;
      }

      .tips {
        height: 22px;
        line-height: 22px;
        color: $title2;
        margin-right: 8px;
      }

      .file {
        color: $baseColor;
        margin-right: 8px;
      }

      svg {
        width: 14px;
        height: 14px;
        color: $baseColor;
        cursor: pointer;
      }
    }

    .feed-back {
      display: flex;
      height: 20px;
      margin-top: 8px;
      margin-left: 48px;

      .reload-box {
        display: flex;
        cursor: pointer;
        align-items: center;
        margin-right: auto;
        color: #5a47e5;
        .reload-text {
          height: 22px;
          line-height: 22px;
        }
      }

      .tools {
        display: flex;
        align-items: center;

        svg {
          margin-left: 16px;
        }
      }

      svg {
        width: 16px !important;
        height: 16px !important;
        cursor: pointer !important;
      }
    }
  }
}

.stop-btn {
  display: flex;
  justify-content: center;
  margin-top: 38px;
  :deep(.ant-btn) {
    width: 92px;
    height: 32px;
    border: 1px solid #e2e2e2;
    color: $title2;
  }
  svg {
    width: 12px;
    height: 12px;
    margin-right: 4px;
  }

  .loading {
    animation: loading 3s infinite;
  }
}

.question-box {
  position: fixed;
  bottom: 28px;
  left: 280px;
  width: calc(100vw - 280px);

  .question {
    width: 75.36%;
    min-width: 900px;
    max-width: 1239px;
    height: 48px;
    margin: 0 auto;
    display: flex;
    align-items: center;

    .download,
    .delete,
    .control {
      cursor: pointer;
      padding: 12px;
      display: flex;
      margin-right: 16px;
      border-radius: 8px;
      background: #ffffff;
      border: 1px solid #e5e5e5;
      color: #666666;

      &:hover {
        border: 1px solid #5a47e5;
        color: #5a47e5;
      }

      svg {
        width: 24px;
        height: 24px;
      }
      &.control-true {
        border: 1px solid #5a47e5;
        color: #5a47e5;
      }
      &.control-false {
        border: 1px solid #e5e5e5;
        color: #666666;
      }
    }

    .send-plane {
      width: 56px;
      height: 36px;
      border-radius: 8px;
      color: #fff;
      background: #5a47e5;

      :deep(.ant-btn-primary) {
        background-color: #5a47e5 !important;
      }
      :deep(.ant-btn-primary:disabled) {
        background-color: #5a47e5 !important;
        color: #fff !important;
        border-color: transparent !important;
      }

      svg {
        width: 24px;
        height: 24px;
      }
    }

    :deep(.ant-input-affix-wrapper) {
      width: 100%;
      max-width: 1108px;
      border-color: #e5e5e5;
      box-shadow: none !important;
      &:hover,
      &:focus,
      &:active {
        border-color: #5a47e5 !important;
        box-shadow: none !important;
      }
    }

    :deep(.ant-input:hover) {
      border-color: $baseColor;
    }
    :deep(.ant-input:focused) {
      border-color: $baseColor;
    }
  }
}

.sourceitem-leave,   // 离开前,进入后透明度是1
.sourceitem-enter-to {
  opacity: 1;
}
.sourceitem-leave-active,
.sourceitem-enter-active {
  transition: opacity 0.5s; //过度是.5s秒
}
.sourceitem-leave-to,
.sourceitem-enter {
  opacity: 0;
}
</style>
<style lang="scss">
@keyframes shake {
  0% {
    transform: rotate(0deg);
  }

  10% {
    transform: rotate(10deg);
  }

  20% {
    transform: rotate(20deg);
  }
  30% {
    transform: rotate(20deg);
  }
  40% {
    transform: rotate(20deg);
  }

  50% {
    transform: rotate(15deg);
  }

  60% {
    transform: rotate(0deg);
  }
  70% {
    transform: rotate(-15deg);
  }
  80% {
    transform: rotate(-30deg);
  }
  90% {
    transform: rotate(-15deg);
  }

  100% {
    transform: rotate(0deg);
  }
}

@keyframes blink {
  from {
    opacity: 0;
  }

  to {
    opacity: 1;
  }
}

@keyframes loading {
  0% {
    transform: rotate(0deg);
  }

  25% {
    transform: rotate(90deg);
  }
  50% {
    transform: rotate(180deg);
  }
  75% {
    transform: rotate(270deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
