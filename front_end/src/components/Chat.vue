<template>
  <HistoryChat
    :observer="observer"
    :observe-dom="observeDom"
    :qa-observe-dom="qaObserveDom"
    :qa-observer="qaObserver"
    :show-loading="showLoading"
    @scrollBottom="scrollBottom"
    @setObserveDom="setObserveDom"
    @setQaObserverDom="setQaObserverDom"
    @clearHistory="clearHistory"
  />
  <div class="container showSider">
    <div class="my-page">
      <div id="chat" ref="chatContainer" class="chat showSider">
        <ul id="chat-ul" ref="scrollDom">
          <li v-for="(item, index) in QA_List" :key="index">
            <div v-if="item.type === 'user'" class="user">
              <img class="avatar" src="../assets/home/avatar.png" alt="头像" />
              <p class="question-text">{{ item.question }}</p>
            </div>
            <div v-else class="ai">
              <img class="avatar" src="../assets/home/ai-avatar.png" alt="头像" />
              <div class="ai-content">
                <div class="ai-right">
                  <!--                  <p-->
                  <!--                    v-if="!item.onlySearch"-->
                  <!--                    class="question-text"-->
                  <!--                    :class="[-->
                  <!--                      !item.source.length ? 'change-radius' : '',-->
                  <!--                      item.showTools ? '' : 'flashing',-->
                  <!--                    ]"-->
                  <!--                    v-html="item.answer"-->
                  <!--                  ></p>-->
                  <p
                    v-if="!item.onlySearch"
                    class="question-text"
                    :class="[
                      !item.source.length && !item?.picList?.length ? 'change-radius' : '',
                      item.showTools ? '' : 'flashing',
                    ]"
                  >
                    <HighLightMarkDown v-if="item.answer" :content="item.answer" />
                    <span v-else>{{ item.answer }}</span>

                    <ChatInfoPanel
                      v-if="Object.keys(item?.itemInfo?.tokenInfo || {}).length"
                      :chat-item-info="item.itemInfo"
                    />
                    <!--                    {{ item.itemInfo }}-->
                  </p>
                  <p
                    v-else-if="item.onlySearch && !item.source.length"
                    class="question-text"
                    :class="[
                      !item.source.length && !item?.picList?.length ? 'change-radius' : '',
                      item.showTools ? '' : 'flashing',
                    ]"
                  >
                    <span v-if="language === 'zh'">未找到信息来源</span>
                    <span v-else>Information source not found</span>
                  </p>

                  <template v-if="item.source.length">
                    <div
                      :class="[
                        'source-total',
                        !showSourceIdxs.includes(index) ? 'source-total-last' : '',
                      ]"
                    >
                      <span v-if="language === 'zh'">
                        <span v-if="item.onlySearch">检索完成，</span>
                        找到了{{ item.source.length }}个信息来源：
                      </span>
                      <span v-else>
                        <span v-if="item.onlySearch">Search completed，</span>
                        Found {{ item.source.length }} source of information
                      </span>
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
                        <p v-show="sourceItem.file_name" class="control">
                          <span class="tips">{{ common.dataSource }}{{ sourceIndex + 1 }}:</span>
                          <a
                            v-if="sourceItem.file_id.startsWith('http')"
                            :href="sourceItem.file_id"
                            target="_blank"
                          >
                            {{ sourceItem.file_name }}
                          </a>
                          <span
                            v-else
                            :class="[
                              'file',
                              checkFileType(sourceItem.file_name) ? 'filename-active' : '',
                            ]"
                            @click="handleChatSource(sourceItem)"
                          >
                            {{ sourceItem.file_name }}
                          </span>
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
              </div>
            </div>
          </li>
        </ul>
      </div>
      <div v-if="showLoading" class="stop-btn">
        <a-button @click="stopChat">
          <template #icon>
            <SvgIcon name="stop" :class="showLoading ? 'loading' : ''"></SvgIcon>
          </template>
          {{ common.stop }}
        </a-button>
      </div>
      <div class="question-box">
        <div class="question">
          <div class="send-box">
            <a-textarea
              v-model:value="question"
              class="send-textarea"
              max-length="200"
              :bordered="false"
              :placeholder="common.problemPlaceholder"
              :auto-size="{ minRows: 4, maxRows: 8 }"
              @pressEnter="send"
            />
            <div class="send-action">
              <a-popover placement="topLeft">
                <template #content>{{ common.chatToPic }}</template>
                <span
                  :class="['download', showLoading ? 'isPreventClick' : '']"
                  @click="downloadChat"
                >
                  <SvgIcon name="chat-download" />
                </span>
              </a-popover>
              <a-popover>
                <template #content>{{ common.clearChat }}</template>
                <span :class="['delete', showLoading ? 'isPreventClick' : '']" @click="deleteChat">
                  <SvgIcon name="chat-delete" />
                </span>
              </a-popover>
              <a-popover>
                <template #content>{{ common.modelSettingTitle }}</template>
                <span class="setting" @click="handleModalChange(true)">
                  <SvgIcon name="chat-setting" />
                </span>
              </a-popover>
              <a-button type="primary" :disabled="showLoading" shape="circle" @click="send">
                <SvgIcon name="sendplane" />
              </a-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <ChatSettingDialog />
  <DefaultModal :content="content" :confirm-loading="confirmLoading" @ok="confirm" />
  <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
    <a-float-button class="scroll-btn" type="primary" @click="scrollBottom">
      <template #icon>
        <SvgIcon name="scroll" />
      </template>
    </a-float-button>
  </a-config-provider>
</template>
<script lang="ts" setup>
import { apiBase } from '@/services';
import { IChatItem } from '@/utils/types';
import { useClipboard, useThrottleFn } from '@vueuse/core';
import { message } from 'ant-design-vue';
import SvgIcon from './SvgIcon.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useChat } from '@/store/useChat';
import { useChatSource } from '@/store/useChatSource';
import { Typewriter } from '@/utils/typewriter';
import DefaultModal from './DefaultModal.vue';
import html2canvas from 'html2canvas';
import urlResquest, { userId } from '@/services/urlConfig';
import { getLanguage } from '@/language';
import { useLanguage } from '@/store/useLanguage';
import { ChatInfoClass, resultControl } from '@/utils/utils';
import ChatSettingDialog from '@/components/ChatSettingDialog.vue';
import HistoryChat from '@/components/Home/HistoryChat.vue';
import { useHomeChat } from '@/store/useHomeChat';
import HighLightMarkDown from '@/components/HighLightMarkDown.vue';
import { useChatSetting } from '@/store/useChatSetting';
// import ChatInfoPanel from '@/components/ChatInfoPanel.vue';

const common = getLanguage().common;

const typewriter = new Typewriter((str: string) => {
  if (str) {
    QA_List.value[QA_List.value.length - 1].answer += str || '';
    console.log(QA_List.value);
  }
});

const { selectList, knowledgeBaseList } = storeToRefs(useKnowledgeBase());
const { QA_List, chatId, pageId, qaPageId, historyList } = storeToRefs(useHomeChat());
const { chatSettingFormActive } = storeToRefs(useChatSetting());
const { copy } = useClipboard();
const { addHistoryList, updateHistoryList, addChatList, clearChatList } = useHomeChat();
const { setChatSourceVisible, setSourceType, setSourceUrl, setTextContent } = useChatSource();
const { language } = storeToRefs(useLanguage());
declare module _czc {
  const push: (array: any) => void;
}

// ai下面的ChatInfoPanel
// const hasTokenInfo = item => {
//   return Object.keys(item?.itemInfo?.tokenInfo || {}).length > 0;
// };

//当前问的问题
const question = ref('');

//问答的上下文
const history = computed(() => {
  const context = chatSettingFormActive.value.context;
  if (context === 0) return [];
  const usefulChat = QA_List.value.filter(item => item.type === 'ai');
  const historyChat = context === 11 ? usefulChat : usefulChat.slice(-context);
  return historyChat.map(item => [item.question, item.answer]);
});

//当前是否回答中
const showLoading = ref(false);

const showSourceIdxs = ref([]);

// 被监听的元素
const observeDom = ref(null);

// 问答列表被监听的元素
const qaObserveDom = ref(null);

//取消请求用
let ctrl: AbortController;

const chatContainer = ref(null);

const scrollDom = ref(null);

const scrollBottom = () => {
  nextTick(() => {
    scrollDom.value?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
  });
};

// 创建 Intersection Observer 对象
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    // 判断元素是否在可视范围内
    if (entry.isIntersecting) {
      console.log('entry.isIntersecting');
      pageId.value++;
      // getHistoryList(pageId.value);
    }
  });
});

// 问答观察者
const qaObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    // 判断元素是否在可视范围内
    if (entry.isIntersecting) {
      console.log('qa entry.isIntersecting');
      qaPageId.value++;
      // getChatDetail(qaPageId.value);
      // getChatDetail();
    }
  });
});

onMounted(() => {
  // getHistoryList(pageId.value);
  // getPrivatizationInfo();
  console.log('------chatId', chatId.value);
  scrollBottom();
});

onBeforeUnmount(() => {
  if (observeDom.value) {
    observer.unobserve(observeDom.value);
  }
  if (qaObserveDom.value) {
    qaObserver.unobserve(qaObserveDom.value);
  }
});

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
  console.log('QALIST------', QA_List.value);
  scrollBottom();
};

const addAnswer = (question: string) => {
  QA_List.value.push({
    answer: '',
    question,
    onlySearch: chatSettingFormActive.value.capabilities.onlySearch,
    type: 'ai',
    copied: false,
    like: false,
    unlike: false,
    source: [],
    showTools: false,
  });
};

const chatInfoClass = new ChatInfoClass();

const setObserveDom = value => {
  observeDom.value = value;
};

const setQaObserverDom = value => {
  qaObserveDom.value = value;
};

const updateChat = (title: string, chatId: number, knowledgeListSelect) => {
  try {
    updateHistoryList(title, chatId, knowledgeListSelect);
  } catch (e) {
    message.error(e.msg || '更新对话失败');
  }
};

function checkKbSelect() {
  if (!selectList.value.length) {
    return;
  }
  // 删除知识库时不会删除对话里表中已经选中的知识库id  所以每次问答前都要校验一下这个知识库id还存不存在
  const list = [];
  selectList.value.forEach(kbId => {
    if (knowledgeBaseList.value.some(item => item.kb_id === kbId)) {
      list.push(kbId);
    }
  });
  selectList.value = list;
  // 如果当前对话选中的知识库有变化 就更新一下
  historyList.value.forEach(item => {
    if (
      chatId.value !== null &&
      item.historyId === chatId.value &&
      item.kbIds.join('') !== selectList.value.join('')
    ) {
      updateChat(item.title, item.historyId, selectList.value);
    }
  });
}

const stopChat = () => {
  if (ctrl) {
    ctrl.abort();
  }
  typewriter.done();
  showLoading.value = false;
  QA_List.value[QA_List.value.length - 1].showTools = true;
};

// 问答前处理 判断创建对话
const beforeSend = title => {
  try {
    console.log('chat-title=', title);
    // 判断需不需要新建对话, 为null直接跳出
    if (chatId.value !== null) return;
    if (title.length > 100) {
      title = title.substring(0, 100);
    }
    // 当前对话id为新建的historyId
    chatId.value = addHistoryList(title);
    updateChat(title, chatId.value, selectList.value);
  } catch (e) {
    message.error(e.msg || '创建对话失败');
  }
};

//发送问答消息
const send = () => {
  if (!question.value.trim().length) {
    return;
  }
  if (showLoading.value) {
    message.warn('正在聊天中...请等待结束');
    return;
  }
  if (!checkChatSetting()) {
    message.error('模型设置错误，请先检查模型配置');
    return;
  }
  checkKbSelect();
  if (!selectList.value.length) {
    return message.warning(common.chooseError);
  } else {
    // 校验选中的知识库
    message.info({
      content:
        common.type === 'zh'
          ? `已选择 ${selectList.value.length} 个知识库进行问答`
          : ` ${selectList.value.length} knowledge base has been selected`,
      icon: ' ',
    });
  }
  const q = question.value;
  beforeSend(q);
  question.value = '';
  addQuestion(q);
  // 更新最大的chatList
  addChatList(chatId.value, QA_List.value);
  showLoading.value = true;
  ctrl = new AbortController();

  fetchEventSource(apiBase + '/local_doc_qa/local_doc_chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: ['text/event-stream', 'application/json'],
    },
    openWhenHidden: true,
    body: JSON.stringify({
      user_id: userId,
      kb_ids: selectList.value,
      history: history.value,
      question: q,
      streaming: chatSettingFormActive.value.capabilities.onlySearch === false,
      networking: chatSettingFormActive.value.capabilities.networkSearch,
      product_source: 'saas',
      rerank: chatSettingFormActive.value.capabilities.rerank,
      only_need_search_results: chatSettingFormActive.value.capabilities.onlySearch,
      hybrid_search: chatSettingFormActive.value.capabilities.mixedSearch,
      max_token: chatSettingFormActive.value.maxToken,
      api_base: chatSettingFormActive.value.apiBase,
      api_key: chatSettingFormActive.value.apiKey,
      model: chatSettingFormActive.value.apiModelName,
      api_context_length: chatSettingFormActive.value.apiContextLength,
      chunk_size: chatSettingFormActive.value.chunkSize,
      top_p: chatSettingFormActive.value.top_P,
      temperature: chatSettingFormActive.value.temperature,
    }),
    signal: ctrl.signal,
    onopen(e: any) {
      console.log('open', e);
      // 模型配置添加进去
      chatInfoClass.addChatSetting(chatSettingFormActive.value);
      if (e.ok && e.headers.get('content-type') === 'text/event-stream') {
        // addAnswer(question.value);
        // question.value = '';
        addAnswer(q);
        typewriter.start();
      } else if (e.headers.get('content-type') === 'application/json') {
        // showLoading.value = false;
        // return e
        //   .json()
        //   .then(data => {
        //     message.error(data?.msg || '出错了,请稍后刷新重试。');
        //   })
        //   .catch(() => {
        //     message.error('出错了,请稍后刷新重试。');
        //   }); // 将响应解析为 JSON
        addAnswer(q);
      }
    },
    onmessage(msg: { data: string }) {
      console.log('message', msg);
      const res: any = JSON.parse(msg.data);
      if (res?.code == 200 && res?.response && res.msg === 'success') {
        // QA_List.value[QA_List.value.length - 1].answer += res.result.response;
        // typewriter.add(res?.response.replaceAll('\n', '<br/>'));
        typewriter.add(res?.response);
        scrollBottom();
      } else {
        const timeObj = res.time_record.time_usage;
        delete timeObj['retriever_search_by_milvus'];
        chatInfoClass.addTime(res.time_record.time_usage);
        chatInfoClass.addToken(res.time_record.token_usage);
        chatInfoClass.addDate(Date.now());
      }

      if (res?.source_documents?.length) {
        QA_List.value[QA_List.value.length - 1].source = res?.source_documents;
      }

      // if (res?.history.length) {
      //   history.value = res?.history;
      // }
    },
    onclose(e: any) {
      console.log('close', e);
      console.log(e);
      typewriter.done();
      ctrl.abort();
      showLoading.value = false;
      QA_List.value[QA_List.value.length - 1].showTools = true;
      // 将chat info添加进回答中
      QA_List.value.at(-1).itemInfo = chatInfoClass.getChatInfo();
      // 更新最大的chatList
      addChatList(chatId.value, QA_List.value);
      nextTick(() => {
        scrollBottom();
      });
      console.log(QA_List.value);
    },
    onerror(err: any) {
      console.log('error', err);
      typewriter?.done();
      ctrl?.abort();
      showLoading.value = false;
      QA_List.value[QA_List.value.length - 1].showTools = true;
      message.error(err.msg || '出错了');
      // 更新最大的chatList
      addChatList(chatId.value, QA_List.value);
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
const confirmLoading = ref(false);
const content = ref('');
const type = ref('');
const downloadChat = () => {
  if (showLoading.value) return;
  type.value = 'download';
  showModal.value = true;
  content.value = common.saveTip;
};

const deleteChat = () => {
  if (showLoading.value) return;
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
    // history.value = [];
    clearChatList(chatId.value);
    chatId.value = null;
    QA_List.value = [];
  }
  type.value = '';
  content.value = '';
  confirmLoading.value = false;
  showModal.value = false;
};

// 模型设置弹窗相关
const { showSettingModal } = storeToRefs(useChat());

const handleModalChange = newVal => {
  showSettingModal.value = newVal;
};

// 模型配置是否正确
const checkChatSetting = () => {
  return !!(
    chatSettingFormActive.value.apiKey &&
    chatSettingFormActive.value.apiBase &&
    chatSettingFormActive.value.apiModelName
  );
};

// 检查信息来源的文件是否支持窗口化渲染
let supportSourceTypes = ['pdf', 'docx', 'xlsx', 'txt', 'jpg', 'png', 'jpeg'];
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
    console.log('queryFile', res);
    const suffix = file.file_name.split('.').pop();
    const b64Type = getB64Type(suffix);
    console.log('b64Type', b64Type);
    setSourceType(suffix);
    setSourceUrl(`data:${b64Type};base64,${res.base64_content}`);
    if (suffix === 'txt') {
      const decodedTxt = atob(res.base64_content);
      const correctStr = decodeURIComponent(escape(decodedTxt));
      console.log('decodedTxt', correctStr);
      setTextContent(correctStr);
      setChatSourceVisible(true);
    } else {
      setChatSourceVisible(true);
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
  'image/jpeg',
  'image/png',
  'image/jpeg',
];

function getB64Type(suffix) {
  const index = supportSourceTypes.indexOf(suffix);
  return b64Types[index];
}

// 清空多轮问答历史
function clearHistory() {
  // 先注了，不知道这方法干啥用的，也不敢动
  console.log('清空');
  // history.value = [];
}

scrollBottom();
</script>

<style lang="scss" scoped>
.container {
  // padding-top: 16px;
  height: calc(100%);
  // margin-top: 65px;

  &.showSider {
    height: calc(100vh - 64px - 64px);
  }
}

.my-page {
  position: relative;
  height: 100%;
  margin: 0 auto;
  //border-radius: 12px 0 0 0;
  //border-top-color: #26293b;
  display: flex;
  flex-direction: column;
  background: #f3f6fd;
  overflow: hidden;
}

.chat {
  margin: 0 auto;
  //width: 75.36%;
  width: 50%;
  min-width: 900px;
  max-width: 1239px;
  padding: 28px 0 0 0;
  //height: calc(100vh - 54px - 48px - 28px - 28px - 32px - 50px);
  flex: 1;
  //overflow: hidden;
  overflow-y: auto;

  &.showSider {
    //height: calc(100vh - 280px);
  }

  #chat-ul {
    background: #f3f6fd;
    overflow: hidden;
  }

  .avatar {
    width: 32px;
    height: 32px;
    margin-right: 16px;
  }

  .user {
    display: flex;
    flex-direction: row-reverse;
    justify-content: flex-start;
    margin-bottom: 16px;

    .avatar {
      margin: 0 0 0 16px;
    }

    .question-text {
      padding: 13px 20px;
      margin-left: 48px;
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      color: #222222;
      background: #e9e1ff;
      border-radius: 12px;
      word-wrap: break-word;
    }
  }

  .ai {
    margin: 16px 0 28px 0;
    display: flex;

    .ai-content {
      display: flex;
      flex-direction: column;
      padding-right: 48px;
      min-width: 20%;

      .question-text {
        flex: 1;
        padding: 13px 20px;
        font-size: 14px;
        font-weight: normal;
        line-height: 22px;
        color: $title1;
        background: #fff;
        border-radius: 12px 12px 0 0;
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
        border-radius: 12px;
      }
    }

    .source-total {
      padding: 13px 20px;
      background: #fff;
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
      border-radius: 0px 0 12px 12px;
    }

    .source-list {
      background: #fff;
      border-radius: 0px 12px 12px 12px;
    }

    .data-source {
      padding: 13px 20px;
      font-size: 14px;
      line-height: 22px;
      color: $title1;

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

      .filename-active {
        color: #5a47e5;
        text-decoration: underline;
        cursor: pointer;
      }

      svg {
        width: 14px;
        height: 14px;
        color: $baseColor;
        cursor: pointer;
      }

      a {
        color: #5a47e5;
        text-decoration: underline;
        cursor: pointer;
      }
    }

    .feed-back {
      display: flex;
      height: 20px;
      margin-top: 8px;

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
  margin: 18px 0;

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
  //position: fixed;
  //bottom: 28px;
  //left: 280px;
  width: 100%;
  margin-bottom: 30px;

  .question {
    //width: 75.36%;
    width: 60%;
    //height: 48px;
    margin: 0 auto;
    display: flex;
    align-items: center;

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

    :deep(.ant-input:focus) {
      border-color: $baseColor;
    }

    .send-box {
      position: relative;
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      align-items: center;
      background-color: #fff;
      border: 1px solid #d9d9d9;
      border-radius: 18px;

      &:hover {
        border-color: $baseColor;
        transition: border-color 0.3s, height 0s;
      }

      &:not(:hover) {
        border-color: #d9d9d9;
        transition: border-color 0.3s;
      }

      &:focus {
        box-shadow: 0 0 0 2px rgba(5, 145, 255, 0.1);
      }

      .send-textarea {
        //position: absolute;
        //bottom: 0;
        min-height: 42px;
        line-height: 25px;
        padding: 11px 15px;
        display: flex;
        align-items: center;
        font-size: 14px;
        border-radius: 18px;
      }
    }

    .send-action {
      width: 100%;
      height: 40px;
      padding-right: 10px;
      display: flex;
      justify-content: flex-end;
      align-items: center;
      color: #fff;
      z-index: 101;

      .isPreventClick {
        cursor: not-allowed !important;
      }

      .download,
      .delete,
      .setting {
        cursor: pointer;
        padding: 8px;
        display: flex;
        margin-right: 16px;
        border-radius: 50%;
        background: #ffffff;
        //border: 1px solid #e5e5e5;
        color: #666666;

        &:hover {
          //border: 1px solid #5a47e5;
          background-color: #e5e5e5;
          color: #5a47e5;
        }

        svg {
          width: 18px;
          height: 18px;
        }
      }

      :deep(.ant-btn-primary) {
        width: 36px;
        height: 36px;
        padding: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(300deg, #7b5ef2 1%, #c383fe 97%);
      }

      :deep(.ant-btn-primary:disabled) {
        height: 36px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(300deg, #7b5ef2 1%, #c383fe 97%);
        color: #fff !important;
        border-color: transparent !important;
      }

      svg {
        width: 24px;
        height: 24px;
      }
    }
  }
}

.scroll-btn {
  inset-inline-end: 17%;
  inset-block-end: 18%;

  svg {
    width: 20px;
    height: 20px;
    margin-top: 5px;
  }
}

.sourceitem-leave, // 离开前,进入后透明度是1
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
