<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-22 16:10:06
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 16:56:13
 * @FilePath: front_end/src/views/QuickStart/index.vue
 * @Description: 快速开始，每个对话对应一个知识库（自动创建），传文件自动放入该对话对应的知识库
 -->
<template>
  <div class="container">
    <OptionList v-if="showDefault === pageStatus.optionlist" />
    <div v-else class="my-page">
      <div id="chat" ref="chatContainer" class="chat">
        <ul id="chat-ul" ref="scrollDom">
          <li v-for="(item, index) in QA_List" :key="index">
            <div v-if="item.type === 'user'" class="user">
              <img class="avatar" src="@/assets/home/avatar.png" alt="头像" />
              <div class="question-inner">
                <p class="question-text">{{ item.question }}</p>
                <div v-if="item.fileDataList.length" class="file-list-box">
                  <FileBlock
                    v-for="file of item.fileDataList"
                    :key="file.file.lastModified"
                    :file-data="file"
                    :kb-id="kbId"
                    :chat-id="chatId"
                    status="send"
                  />
                </div>
              </div>
            </div>
            <div v-else class="ai">
              <img class="avatar" src="@/assets/home/ai-avatar.png" alt="头像" />
              <div class="ai-content">
                <div class="ai-right">
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
                        <Transition name="sourceItem">
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
                      <SvgIcon name="reload" />
                      <span class="reload-text">{{ common.regenerate }}</span>
                    </div>
                    <div class="tools">
                      <SvgIcon
                        :style="{
                          color: item.copied ? '#4D71FF' : '',
                        }"
                        name="copy"
                        @click="myCopy(item)"
                      />
                      <SvgIcon
                        :style="{
                          color: item.like ? '#4D71FF' : '',
                        }"
                        name="like"
                        @click="like(item, $event)"
                      />
                      <SvgIcon
                        :style="{
                          color: item.unlike ? '#4D71FF' : '',
                        }"
                        name="unlike"
                        @click="unlike(item)"
                      />
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
          <div v-if="fileBlockArr.length" class="file-list-box">
            <FileBlock
              v-for="file of fileBlockArr"
              :key="file.file_id"
              :file-data="file"
              :kb-id="kbId"
              :chat-id="chatId"
              status="toBeSend"
              @deleteFile="deleteFile"
            />
          </div>
          <div class="send-box">
            <a-textarea
              v-model:value="question"
              class="send-textarea"
              max-length="200"
              :bordered="false"
              :placeholder="common.problemPlaceholder"
              :auto-size="{ minRows: 4, maxRows: 8 }"
              @keydown="textKeydownHandle"
            />
            <div class="send-action">
              <a-popover>
                <template #content>
                  {{ kbId ? common.chatUpload : common.chatUploadNoKbId }}
                </template>
                <span
                  :class="['question-icon', showLoading || !kbId ? 'isPreventClick' : '']"
                  @click="uploadFile"
                >
                  <SvgIcon name="chat-upload" />
                </span>
              </a-popover>
              <a-popover>
                <template #content>{{ common.chatToPic }}</template>
                <span
                  :class="['question-icon', showLoading ? 'isPreventClick' : '']"
                  @click="downloadChat"
                >
                  <SvgIcon name="chat-download" />
                </span>
              </a-popover>
              <a-popover>
                <template #content>{{ common.modelSettingTitle }}</template>
                <span class="question-icon" @click="handleModalChange(true)">
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
      <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
        <a-float-button class="scroll-btn" type="primary" @click="scrollBottom">
          <template #icon>
            <SvgIcon name="scroll" />
          </template>
        </a-float-button>
      </a-config-provider>
    </div>
  </div>
  <ChatSettingDialog ref="chatSettingForDialogRef" />
  <DefaultModal :content="content" :confirm-loading="confirmLoading" @ok="confirm" />
  <FileUploadDialog :dialog-type="1" />
</template>

<script setup lang="ts">
import HighLightMarkDown from '@/components/HighLightMarkDown.vue';
import SvgIcon from '@/components/SvgIcon.vue';
import { getLanguage } from '@/language';
import { Typewriter } from '@/utils/typewriter';
import { useClipboard, useThrottleFn } from '@vueuse/core';
import { IChatItem, IFileListItem } from '@/utils/types';
import { message } from 'ant-design-vue';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { apiBase } from '@/services';
import urlResquest, { userId } from '@/services/urlConfig';
import { useChat } from '@/store/useChat';
import html2canvas from 'html2canvas';
import { ChatInfoClass, resultControl } from '@/utils/utils';
import { useChatSource } from '@/store/useChatSource';
import { useLanguage } from '@/store/useLanguage';
import { useQuickStart } from '@/store/useQuickStart';
import DefaultModal from '@/components/DefaultModal.vue';
import ChatSettingDialog from '@/components/ChatSettingDialog.vue';
import { pageStatus } from '@/utils/enum';
import OptionList from '@/components/OptionList.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import FileBlock from '@/views/QuickStart/children/FileBlock.vue';
import { useUploadFiles } from '@/store/useUploadFiles';
import { useChatSetting } from '@/store/useChatSetting';
import FileUploadDialog from '@/components/FileUploadDialog.vue';
import ChatInfoPanel from '@/components/ChatInfoPanel.vue';

const { common, home } = getLanguage();

const { copy } = useClipboard();
const { QA_List, chatId, kbId, kbIdCopy, showLoading } = storeToRefs(useQuickStart());
const {
  addHistoryList,
  updateHistoryList,
  addChatList,
  clearChatList,
  getHistoryById,
  renameHistory,
  addFileToBeSendList,
} = useQuickStart();
const { setChatSourceVisible, setSourceType, setSourceUrl, setTextContent } = useChatSource();
const { chatSettingFormActive } = storeToRefs(useChatSetting());
const { showDefault } = storeToRefs(useKnowledgeBase());
const { setModalVisible, setModalTitle } = useKnowledgeModal();
const { uploadFileListQuick } = storeToRefs(useUploadFiles());
const { initUploadFileListQuick } = useUploadFiles();
const { language } = storeToRefs(useLanguage());

declare module _czc {
  const push: (array: any) => void;
}

// 当前对话的historyList
const historyList = computed(() => {
  return getHistoryById(chatId.value);
});

const typewriter = new Typewriter((str: string) => {
  if (str) {
    QA_List.value[QA_List.value.length - 1].answer += str || '';
  }
});
//当前问的问题
const question = ref('');

//问答的上下文
const history = computed(() => {
  const context = chatSettingFormActive.value.context;
  if (context === 0) return [];
  const usefulChat = QA_List.value.filter(item => item.type === 'ai');
  const historyChat = context === 1 ? usefulChat : usefulChat.slice(-context);
  return historyChat.map(item => [item.question, item.answer]);
});

const showSourceIdxs = ref([]);

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

// fileBlock的数组，为缓存拿的和这次上传的联合起来的
const fileBlockArr = computed<IFileListItem[]>(() => {
  const arr = historyList.value?.fileToBeSendList.concat(uploadFileListQuick?.value) || [];
  return arr.filter(item => item.file_id.length !== 0);
});

onMounted(() => {
  kbIdCopy.value && (kbId.value = kbIdCopy.value);
  scrollBottom();
});

// 切换到知识库或者bot，切换对话的在Sider里面
onBeforeUnmount(() => {
  if (uploadFileListQuick.value.length) {
    addFileToBeSendList(chatId.value, [...uploadFileListQuick.value]);
    initUploadFileListQuick();
  }
});

// 聊天框keydown，不允许enter换行，alt/ctrl/shift/meta(Command或win) + enter可换行
const textKeydownHandle = e => {
  // 首先检查是否按下 Enter 键
  if (e.keyCode === 13) {
    if (e.altKey || e.ctrlKey || e.shiftKey || e.metaKey) {
      question.value += '\n';
    } else {
      send();
    }
    // 放在外面因为shift + enter本来就是换行，会触发两次
    e.preventDefault();
  }
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

// 删除文件
const deleteFile = async (file_id: string, kb_id: string) => {
  console.log(file_id);
  try {
    await resultControl(await urlResquest.deleteFile({ file_ids: [file_id], kb_id }));
    message.success('删除成功');
    const index: number = uploadFileListQuick.value.findIndex(item => item.file_id === file_id);
    if (index !== -1) {
      // 在uploadFileListQuick里面
      uploadFileListQuick.value.splice(index, 1);
    } else {
      // 在本地存储里面
      historyList.value.fileToBeSendList = historyList.value.fileToBeSendList.filter(
        item => item.file_id !== file_id
      );
    }
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
};

const addQuestion = q => {
  QA_List.value.push({
    question: q,
    type: 'user',
    fileDataList: [...fileBlockArr.value],
  });
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

const stopChat = () => {
  if (ctrl) {
    ctrl.abort();
  }
  typewriter.done();
  showLoading.value = false;
  QA_List.value[QA_List.value.length - 1].showTools = true;
};

// 问答前处理 判断创建对话
const beforeSend = async title => {
  try {
    // 判断需不需要新建对话, 为null跳出
    if (chatId.value !== null) {
      // 判断是不是刚新建的对话（title = '未命名对话'）是就重命名知识库名字为第一句话
      if (getHistoryById(chatId.value).title === '未命名对话') {
        renameHistory(chatId.value, title);
      }
      return;
    }
    // 需要新建对话（正常操作不会进入这里，因为点击开启新对话就是新建了）
    if (title.length > 100) {
      title = title.substring(0, 100);
    }
    // 创建知识库
    const res: any = await resultControl(await urlResquest.createKb({ kb_name: title }));
    kbId.value = res.kb_id;
    // 当前对话id为新建的historyId
    chatId.value = addHistoryList(title);
    updateHistoryList(title, chatId.value, kbId.value);
  } catch (e) {
    message.error(e.msg || '创建对话失败');
  }
};

//发送问答消息
const send = async () => {
  if (!question.value.trim().length) {
    return;
  }
  if (showLoading.value) {
    message.warn('正在聊天中...请等待结束');
    return;
  }
  if (!(await checkChatSetting())) {
    message.error('模型设置错误，请先检查模型配置');
    return;
  }
  const q = question.value;
  await beforeSend(q);
  question.value = '';
  addQuestion(q);
  // 更新最大的chatList
  addChatList(chatId.value, QA_List.value);
  initUploadFileListQuick();
  historyList.value.fileToBeSendList = [];
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
      kb_ids: [kbId.value],
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
    async onopen(e: any) {
      console.log('eee', e);
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
        // if (res.code !== 200) {
        //   message.error(data?.msg || '出错了,请稍后刷新重试。');
        // }
        console.log('ssss');
        addAnswer(q);
      }
    },
    onmessage(msg: { data: string }) {
      console.log('onmessage', msg);
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
    },
    onclose() {
      console.log('onclose');
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
    },
    onerror(err: any) {
      console.log('onerror', err);
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

//上传文件 下载 清除 聊天记录相关
const { showModal } = storeToRefs(useChat());
const confirmLoading = ref(false);
const content = ref('');
const type = ref('');

const uploadFile = () => {
  if (!kbId.value) return;
  setModalVisible(true);
  setModalTitle(home.upload);
};

const downloadChat = () => {
  if (showLoading.value) return;
  type.value = 'download';
  showModal.value = true;
  content.value = common.saveTip;
};

const confirm = async () => {
  confirmLoading.value = true;
  if (type.value === 'download') {
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
      await Promise.resolve();
    } catch (e) {
      message.error(e.message || e.msg || '出错了');
    }
  } else if (type.value === 'delete') {
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
const chatSettingForDialogRef = ref<InstanceType<typeof ChatSettingDialog>>();
const checkChatSetting = () => {
  return chatSettingForDialogRef.value.handleOk();
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
    return supportSourceTypes.includes(suffix);
  } else {
    return false;
  }
};

const handleChatSource = file => {
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
    setSourceType(suffix);
    setSourceUrl(`data:${b64Type};base64,${res.base64_content}`);
    if (suffix === 'txt') {
      const decodedTxt = atob(res.base64_content);
      const correctStr = decodeURIComponent(escape(decodedTxt));
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

onBeforeUnmount(() => {
  console.log('unmounted');
  kbIdCopy.value = kbId.value;
  kbId.value = '';
});
</script>

<style lang="scss" scoped>
.container {
  width: 100%;
  height: calc(100vh - 64px);
  background-color: #26293b;
}

.my-page {
  position: relative;
  height: 100%;
  margin: 0 auto;
  border-radius: 12px 0 0 0;
  display: flex;
  flex-direction: column;
  background: #f3f6fd;
  overflow: hidden;
}

.chat {
  margin: 0 auto;
  width: 35%;
  min-width: 500px;
  padding: 28px 0 0 0;
  flex: 1;
  overflow-y: auto;

  #chat-ul {
    padding-bottom: 20px;
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

    .question-inner {
      display: flex;
      flex-direction: column;
      align-items: flex-end;

      .file-list-box {
        padding-right: 5px;
        display: flex;
        justify-content: flex-end;
      }
    }

    .question-text {
      //display: flex;
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
      padding: 0 20px 13px 20px;
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
      border-radius: 0 0 12px 12px;
    }

    .source-list {
      background: #fff;
      border-radius: 0 12px 12px 12px;
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

.file-list-box {
  width: 100%;
  max-height: 160px;
  padding: 10px 20px;
  margin: 0 auto;
  display: flex;
  justify-content: flex-start;
  flex-wrap: wrap;
  gap: 8px;
  overflow-y: auto;
}

.question-box {
  width: 100%;
  margin-bottom: 30px;

  .question {
    width: 40%;
    min-width: 550px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
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

      .question-icon {
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
  inset-inline-end: 20%;
  inset-block-end: 18%;

  svg {
    width: 20px;
    height: 20px;
    margin-top: 5px;
  }
}

.sourceItem-leave, // 离开前,进入后透明度是1
.sourceItem-enter-to {
  opacity: 1;
}

.sourceItem-leave-active,
.sourceItem-enter-active {
  transition: opacity 0.5s; //过度是.5s秒
}

.sourceItem-leave-to,
.sourceItem-enter {
  opacity: 0;
}

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
