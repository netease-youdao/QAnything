<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:43:45
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 14:40:16
 * @FilePath: front_end/src/views/QuickStart/components/FileBlock.vue
 * @Description: 这是默认设置,可以在设置》工具》File Description中进行配置
 -->
<template>
  <div class="contain">
    <div class="file-icon" @click="previewHandle">
      <SvgIcon :name="fileIcon" />
      <div :class="isLoading || fileStatus === 'yellow' ? 'mask' : ''" />
      <div v-if="isLoading" class="loading">
        <LoadingImg />
      </div>
      <div v-if="fileStatus === 'yellow'" class="progress">
        <a-progress type="circle" :percent="fileProgress" :size="25" />
      </div>
    </div>
    <div class="file-info" @click="previewHandle">
      <span class="file-name">{{ fileInfo.fileName }}</span>
      <span v-if="fileStatus === 'green'" class="file-extension">
        {{ fileInfo.fileExtension.toUpperCase() }}, {{ formatFileSize(fileData.bytes) || 0 }}
      </span>
      <span v-else class="file-extension">
        {{ fileStatusMap.get(fileStatus) }}
      </span>
    </div>
    <div v-if="status === 'toBeSend'" class="file-close" @click="cancelFile">
      <SvgIcon name="close" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { IFileListItem } from '@/utils/types';
import SvgIcon from '@/components/SvgIcon.vue';
import { formatFileSize, parseFileName, resultControl } from '@/utils/utils';
import LoadingImg from '@/components/LoadingImg.vue';
import urlResquest from '@/services/urlConfig';
import { useChatSource } from '@/store/useChatSource';
import message from 'ant-design-vue/es/message';

const { setChatSourceVisible, setSourceType, setSourceUrl, setTextContent } = useChatSource();

interface IProps {
  fileData: IFileListItem;
  kbId: string;
  status: 'toBeSend' | 'send';
}

const emit = defineEmits(['deleteFile']);

const props = defineProps<IProps>();

const { fileData, kbId } = toRefs(props);

const isLoading = computed(() => {
  return fileStatus.value === 'gray';
});

const fileInfo = computed(() => {
  return parseFileName(fileData.value.file_name);
});

const fileIcon = computed(() => {
  if (fileStatus.value === 'red') {
    return 'file-error';
  } else if (fileStatus.value === 'green') {
    return 'file-' + iconMap.get(fileInfo.value.fileExtension);
  } else {
    return 'file-unknown';
  }
});

const fileStatus = computed(() => {
  return fileData.value.status;
});

// 后缀 -> icon的map
const iconMap: Map<string, string> = new Map([
  ['md', 'txt'],
  ['txt', 'txt'],
  ['pdf', 'pdf'],
  ['jpg', 'img'],
  ['png', 'img'],
  ['jpeg', 'img'],
  ['docx', 'docx'],
  ['xlsx', 'xlsx'],
  ['pptx', 'ppt'],
  ['eml', 'eml'],
  ['csv', 'xlsx'],
]);

// 文件当前状态
/**
 * 名字不变，上传中 -> 解析中 -> 后缀名，全程是loading
 */
const fileStatusMap = new Map([
  ['gray', '上传中'],
  ['yellow', '解析中'],
  ['green', '解析成功'],
  ['red', '解析失败'],
]);

const fileProgress = ref(0);

const getDetail = () => {
  let timer = ref(null);
  timer.value = setInterval(async () => {
    // soft上传重复名字的时候，传来的是fileId: ''
    if (fileData.value.file_id === '' && fileData.value.status !== 'loading') {
      fileData.value.status = 'red';
    }
    if (fileData.value.status === 'green' || fileData.value.status === 'red') {
      clearInterval(timer.value);
      timer.value = null;
    } else {
      const res = (await resultControl(
        await urlResquest.fileList({
          kb_id: kbId.value,
          file_id: fileData.value.file_id,
        })
      )) as any;
      fileData.value.status = res.details[0]?.status || 'red';
      if (res.details[0]?.status === 'yellow') {
        fileProgress.value = parseInt(res.details[0]?.msg.match(/\d+/)[0], 10);
      }
    }
  }, 1000);
};

const cancelFile = async () => {
  emit('deleteFile', fileData.value.file_id, kbId.value);
};

onMounted(() => {
  getDetail();
});

// 预览
const previewHandle = () => {
  if (fileStatus.value === 'yellow' || fileStatus.value === 'green') {
    handleChatSource(fileData.value);
  } else {
    message.warn('解析中、解析完成才可以预览');
  }
};

// 检查信息来源的文件是否支持窗口化渲染
let supportSourceTypes = ['pdf', 'docx', 'xlsx', 'txt', 'md', 'jpg', 'png', 'jpeg'];
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
    setSourceUrl(`data:${b64Type};base64,${res.file_base64}`);
    if (suffix === 'txt' || suffix === 'md') {
      const decodedTxt = atob(res.file_base64);
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
  'text/markdown',
  'image/jpeg',
  'image/png',
  'image/jpeg',
];

function getB64Type(suffix) {
  const index = supportSourceTypes.indexOf(suffix);
  return b64Types[index];
}
</script>

<style lang="scss" scoped>
.contain {
  position: relative;
  width: calc((100% - 16px) / 3);
  min-width: 160px;
  height: 62px;
  padding: 8px;
  display: flex;
  align-items: center;
  border-radius: 12px;
  background-color: #fff;
  box-shadow: 0 8px 16px 0 #0000000d;
  cursor: pointer;
  z-index: 100;

  .file-icon {
    position: relative;
    display: flex;
    align-items: center;

    svg {
      width: 40px;
      height: 40px;
    }

    .loading {
      position: absolute;
      left: 6px;
      width: 20px;
      height: 20px;
    }

    .progress {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }

    .mask {
      position: absolute;
      width: 40px;
      height: 40px;
      background-color: rgba(255, 255, 255, 0.7);
    }
  }

  .file-info {
    margin-left: 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;

    .file-name {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      word-break: keep-all;
    }

    span {
      min-height: 20px;
      font-family: PingFang SC;
      font-weight: 500;
      font-size: 12px;
      line-height: 20px;
    }
  }

  .file-close {
    position: absolute;
    top: 0;
    right: 0;
    transform: translate(20%, -20%);
    cursor: pointer;
    z-index: 101;

    svg {
      width: 16px;
      height: 16px;
    }
  }
}
</style>
