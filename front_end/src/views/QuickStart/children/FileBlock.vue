<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:43:45
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 14:40:16
 * @FilePath: front_end/src/views/QuickStart/children/FileBlock.vue
 * @Description: 这是默认设置,可以在设置》工具》File Description中进行配置
 -->
<template>
  <div class="contain">
    <div class="file-icon">
      <SvgIcon :name="fileIcon" />
      <div v-if="isLoading" class="loading">
        <LoadingImg />
      </div>
    </div>
    <div class="file-info">
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

interface IProps {
  fileData: IFileListItem;
  kbId: string;
  status: 'toBeSend' | 'send';
}

const emit = defineEmits(['deleteFile']);

const props = defineProps<IProps>();

const { fileData, kbId } = toRefs(props);

const isLoading = computed(() => {
  return fileStatus.value !== 'green' && fileStatus.value !== 'red';
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
      console.log(res);
      fileData.value.status = res.details[0]?.status || 'red';
    }
  }, 2000);
};

const cancelFile = async () => {
  emit('deleteFile', fileData.value.file_id, kbId.value);
};

onMounted(() => {
  getDetail();
});
</script>
z
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

  .file-icon {
    display: flex;
    align-items: center;

    svg {
      width: 40px;
      height: 40px;
    }

    .loading {
      position: absolute;
      left: 9px;
      width: 20px;
      height: 20px;
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

    svg {
      width: 16px;
      height: 16px;
    }
  }
}
</style>
