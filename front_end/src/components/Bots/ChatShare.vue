<template>
  <div class="bots-chat-container">
    <div class="header">
      <img src="@/assets/bots/bot-avatar.png" alt="avatar" />
      {{ botInfo.name }}
    </div>
    <div id="chat" class="chat">
      <ul id="chat-ul" ref="scrollDom">
        <div class="ai">
          <div class="content">
            <img class="avatar" src="@/assets/home/ai-avatar.png" alt="头像" />
            <p class="question-text" v-html="botInfo.welcomeMessage"></p>
          </div>
        </div>
        <li v-for="(item, index) in QA_List" :key="index">
          <div v-if="item.type === 'user'" class="user">
            <img class="avatar" src="@/assets/home/avatar.png" alt="头像" />
            <p class="question-text">{{ item.question }}</p>
          </div>
          <div v-else class="ai">
            <div class="content">
              <img class="avatar" src="@/assets/home/ai-avatar.png" alt="头像" />
              <p
                class="question-text"
                :class="[
                  !item.source.length ? 'change-radius' : '',
                  item.showTools ? '' : 'flashing',
                ]"
                v-html="item.answer"
              ></p>
            </div>
            <template v-if="item?.picList?.length">
              <div
                v-for="(picItem, picIndex) in item.picList"
                :key="picItem + picIndex"
                class="data-picList"
              >
                <a-image :width="150" :src="picItem" class="responsive-image" />
              </div>
            </template>
            <template v-if="item.source.length">
              <div
                :class="[
                  'source-total',
                  !showSourceIdxs.includes(index) ? 'source-total-last' : '',
                ]"
              >
                <span v-if="language === 'zh'">找到了{{ item.source.length }}个信息来源：</span>
                <span v-else>Found {{ item.source.length }} source of information</span>
                <SvgIcon
                  v-show="!showSourceIdxs.includes(index)"
                  name="down"
                  @click="showSourceList(index)"
                />
                <SvgIcon
                  v-show="showSourceIdxs.includes(index)"
                  name="up"
                  @click="hideSourceList(index)"
                />
              </div>
              <div v-show="showSourceIdxs.includes(index)" class="source-list">
                <div
                  v-for="(sourceItem, sourceIndex) in item.source"
                  :key="sourceIndex"
                  class="data-source"
                >
                  <p v-show="sourceItem.fileName" class="control">
                    <span class="tips">{{ common.dataSource }}{{ sourceIndex + 1 }}:</span
                    ><span class="file">{{ sourceItem.fileName }}</span>
                    <SvgIcon
                      v-show="sourceItem.showDetailDataSource"
                      name="iconup"
                      @click="hideDetail(item, sourceIndex)"
                    />
                    <SvgIcon
                      v-show="!sourceItem.showDetailDataSource"
                      name="icondown"
                      @click="showDetail(item, sourceIndex)"
                    />
                  </p>
                  <Transition name="sourceitem">
                    <div v-show="sourceItem.showDetailDataSource" class="source-content">
                      <p v-html="sourceItem.content?.replaceAll('\n', '<br/>')"></p>
                      <p class="score">
                        <span class="tips">{{ common.correlation }}</span
                        >{{ sourceItem.score }}
                      </p>
                    </div>
                  </Transition>
                </div>
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
        <div v-show="showLoading" ref="stopBtn" class="stop-btn">
          <a-button @click="stopChat">
            <template #icon>
              <SvgIcon name="stop" :class="showLoading ? 'loading' : ''"></SvgIcon> </template
            >{{ common.stop }}</a-button
          >
        </div>
      </ul>
      <div class="question-box">
        <div class="question">
          <a-popover v-if="chatType === 'share'" placement="topLeft">
            <template #content>
              <p v-if="control">{{ bots.multiTurnConversation2 }}</p>
              <p v-else>{{ bots.multiTurnConversation1 }}</p>
            </template>
            <span :class="['control', `control-${control}`]">
              <SvgIcon name="chat-control" @click="controlChat" />
            </span>
          </a-popover>

          <span v-if="chatType === 'share'" class="download" @click="downloadChat">
            <SvgIcon name="chat-download" />
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
    <div v-if="!botInfo.kbBindList || !botInfo.kbBindList.length" class="mask">
      <img src="@/assets/bots/lock.png" alt="icon" />
      <p>{{ bots.bindKbtoPreview }}</p>
    </div>
  </div>
  <DefaultModal :content="content" :confirm-loading="confirmLoading" @ok="confirm" />
</template>
<script lang="ts" setup>
import { defineProps, defineEmits } from 'vue';
import { IChatItem } from '@/utils/types';
import { useThrottleFn, useClipboard } from '@vueuse/core';
import { message } from 'ant-design-vue';
import SvgIcon from '../SvgIcon.vue';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useBotsChat } from '@/store/useBotsChat';
import { useChat } from '@/store/useChat';
import { useSource } from '@/store/useSource';
import { Typewriter } from '@/utils/typewriter';
import DefaultModal from '../DefaultModal.vue';
import html2canvas from 'html2canvas';
import { getLanguage } from '@/language/index';
import routeController from '@/controller/router';
import { getToken, getShareToken } from '@/utils/token';
import { useLanguage } from '@/store/useLanguage';

const props = defineProps({
  chatType: {
    type: String,
    default: 'edit',
  },
  botInfo: {
    type: Object as any,
    default: () => {},
  },
});

const { getCurrentRoute } = routeController();
const common = getLanguage().common;
const bots = getLanguage().bots;
const store = useBotsChat();
const route = getCurrentRoute();

const emits = defineEmits(['botInit']);

const typewriter = new Typewriter((str: string) => {
  if (str) {
    QA_List.value[QA_List.value.length - 1].answer += str || '';
  }
});

const { QA_List } = storeToRefs(useBotsChat());
const { copy } = useClipboard();
const { setSourceVisible, setPdfSrc, setChunks, setPageSizes, setChunkIds, setPageId } =
  useSource();
const { language } = storeToRefs(useLanguage());
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

const showSourceIdxs = ref([]);

//取消请求用
let ctrl: AbortController;

const scrollDom = ref(null);
const stopBtn = ref(null);

const scrollBottom = () => {
  nextTick(() => {
    scrollDom.value.scrollTop = scrollDom.value.scrollHeight;
  });
};

const like = useThrottleFn((item, e) => {
  item.like = !item.like;
  item.unlike = false;
  _czc.push(['_trackEvent', 'qanything', 'share_like_click', '点赞', '', '']);
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
  _czc.push(['_trackEvent', 'qanything', '分享问答页面', '点踩', '', '']);
};

//拷贝
const myCopy = (item: IChatItem) => {
  _czc.push(['_trackEvent', 'qanything', 'share_copy_click', '复制单个回答', '', '']);
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
    picList: null,
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
  _czc.push(['_trackEvent', 'qanything', 'share_send_click', '分享页发送问题', '', '']);
  if (!question.value.length) {
    return;
  }
  const q = question.value;
  question.value = '';
  addQuestion(q);
  if (history.value.length >= 3) {
    history.value = [];
  }
  showLoading.value = true;
  nextTick(() => {
    scrollDom.value?.scrollIntoView(true);
    stopBtn.value?.scrollIntoView();
  });
  const severUrl = import.meta.env.VITE_APP_SERVER_URL;
  ctrl = new AbortController();

  const headers = {
    'Content-Type': 'application/json',
    Accept: ['text/event-stream', 'application/json'],
    'Transfer-Encoding': 'chunked',
    Connection: 'keep-alive',
  };
  if (route.value.name === 'share') {
    if (getShareToken()) {
      const token = getShareToken();
      headers[token.tokenName] = token.tokenValue;
    }
  } else {
    if (getToken()) {
      const token = getToken();
      headers[token.tokenName] = token.tokenValue;
    }
  }

  fetchEventSource(severUrl + '/q_anything/bot/chat_stream', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify({
      kbIds: props.botInfo.kbBindList.map(item => item.kbId),
      history: control.value ? history.value : [],
      question: q,
      botId: props.botInfo.id,
      botPromptSetting: props.botInfo.promptSetting,
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
            console.log(data);
            if (data?.errorCode === 403) {
              emits('botInit');
            } else {
              message.error(data?.msg || '出错了,请稍后刷新重试。');
            }
          })
          .catch(e => {
            console.log(e);
            message.error('出错了,请稍后刷新重试。');
          }); // 将响应解析为 JSON
      }
    },
    onmessage(msg: { data: string }) {
      console.log('message', msg);
      if (msg.data !== '') {
        const data: any = JSON.parse(msg.data);
        const res = data.result;
        console.log(res);
        console.log(res?.response);
        if (res?.response && !res?.history?.length) {
          // QA_List.value[QA_List.value.length - 1].answer += res.result.response;
          typewriter.add(res?.response.replaceAll('\n', '<br/>'));
          scrollBottom();
        }

        if (res?.picList?.length) {
          QA_List.value[QA_List.value.length - 1].picList = res.picList;
        }

        if (res?.source?.length) {
          console.log('res?.source', res?.source);
          store.handleSource(res.source);
        }

        if (res?.history?.length) {
          history.value = res?.history;
        }
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
      console.log('error', err);
      typewriter.done();
      ctrl.abort();
      showLoading.value = false;
      QA_List.value[QA_List.value.length - 1].showTools = true;

      nextTick(() => {
        scrollBottom();
      });
      throw err;
    },
  });
};

const reAnswer = (item: IChatItem) => {
  _czc.push(['_trackEvent', 'qanything', 'share_regenerate_click', '重新生成回答', '', '']);
  console.log('reAnswer');
  question.value = item.question;
  send();
};

//点击查看是否显示详细来源
const showDetail = (item: IChatItem, index) => {
  console.log('showDetail', item);
  if (item.source[index].pdf_source_info) {
    setPdfSrc(item.source[index].pdf_source_info.pdf_nos_url);
    setChunks(item.source[index].chunks);
    setPageSizes(item.source[index].pageSizes);
    setChunkIds(item.source[index].pdf_source_info.chunk_id);
    setPageId(item.source[index].pdf_source_info.page_id);
    setSourceVisible(true);
  } else {
    item.source[index].showDetailDataSource = true;
  }
};

const hideDetail = (item: IChatItem, index) => {
  item.source[index].showDetailDataSource = false;
};

const showSourceList = index => {
  showSourceIdxs.value.push(index);
};

const hideSourceList = index => {
  showSourceIdxs.value = showSourceIdxs.value.filter(item => item !== index);
};

//下载 清除聊天记录相关
const { showModal } = storeToRefs(useChat());
const { clearQAList } = useBotsChat();
const confirmLoading = ref(false);
const content = ref('');
const type = ref('');
const controlChat = () => {
  control.value = !control.value;
};

const downloadChat = () => {
  _czc.push(['_trackEvent', 'qanything', 'share_save_click', '保存会话', '', '']);
  type.value = 'download';
  showModal.value = true;
  content.value = common.saveTip;
};

// const deleteChat = () => {
//   _czc.push(['_trackEvent', 'qanything', 'empty_click', '清空会话', '', '']);
//   type.value = 'delete';
//   showModal.value = true;
//   content.value = common.clearTip;
// };

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
      message.success(bots.downloadSuccessful);
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
_czc.push(['_trackEvent', 'qanything', 'share_conversation_page_show', '分享问答页曝光', '', '']);
</script>

<style lang="scss" scoped>
.bots-chat-container {
  width: 100%;
  height: 100%;
  border-radius: 12px;
  background: #fff;
  font-family: PingFang SC;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
}
.header {
  width: 100%;
  padding: 0.25rem 0;
  font-size: 14px;
  font-weight: 500;
  color: #222222;
  border-top-right-radius: 12px;
  border-top-left-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid #ededed;
  img {
    width: 32px;
    height: 32px;
    margin-right: 8px;
  }
}
.chat {
  // margin: 0 auto;
  // width: 100%;
  height: 100%;
  padding: 0 10%;
  // min-width: 900px;
  // max-width: 1239px;
  // height: calc(100vh - 64px - 22px - 66px - 52px - 48px - 80px);
  overflow: hidden;
  display: flex;
  flex-direction: column;

  #chat-ul {
    flex: auto 1 1;
    overflow-y: auto;
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
        background: #f9f9fc;
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

    .source-total {
      padding: 13px 20px;
      margin-left: 48px;
      font-size: 14px;
      background: #f9f9fc;
      display: flex;
      align-items: center;
      span {
        margin-right: 5px;
      }
      svg {
        width: 16px !important;
        height: 16px !important;
        cursor: pointer !important;
      }
    }
    .source-total-last {
      border-radius: 0px 12px 12px 12px;
    }

    .source-list {
      margin-left: 48px;
      background: #f9f9fc;
      border-radius: 0px 12px 12px 12px;
    }

    .data-source {
      padding: 13px 20px;
      font-size: 14px;
      line-height: 22px;
      color: $title1;

      &:nth-last-of-type(2) {
        border-radius: 0px 0px 12px 12px;
      }

      &:nth-first-of-type(1) {
        border-radius: 0px 12px 12px 12px;
      }
      .control {
        width: 100%;
        overflow-wrap: break-word;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        .file {
          max-width: 100%;
          word-wrap: break-word;
          overflow-wrap: break-word;
        }
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

    .data-picList {
      margin-left: 48px;
      background: #f9f9fc;
      padding: 10px 20px;
      border-radius: 0 0 12px 12px;
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
  width: 100%;
  height: 52px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 10px 0;
  :deep(.ant-btn) {
    width: 92px;
    height: 32px;
    border: 1px solid #e2e2e2;
    color: $title2;
    display: flex;
    justify-content: center;
    align-items: center;
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
  // position: absolute;
  // bottom: 28px;
  // left: 280px;
  width: 100%;
  margin-top: 10px;
  margin-bottom: 20px;

  .question {
    width: 100%;
    // min-width: 900px;
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
    :deep(.ant-input-affix-wrapper) {
      padding: 4px 4px 4px 11px;
    }
  }
}

.mask {
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.2);
  display: flex;
  color: #fff;
  font-size: 16px;
  border-radius: 12px;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: absolute;
  top: 0;
  left: 0;
  img {
    width: 40px;
    height: 40px;
    margin-bottom: 10px;
  }
  p {
    padding: 0 40px;
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

@media (max-width: 1023px) {
  .chat {
    padding: 0 1rem;
  }
}
@media (min-width: 1500px) {
  .chat {
    padding: 0 20%;
  }
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
