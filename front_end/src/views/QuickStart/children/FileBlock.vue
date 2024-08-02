<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:43:45
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-02 18:16:28
 * @FilePath: front_end/src/views/QuickStart/children/FileBlock.vue
 * @Description: 这是默认设置,可以在设置》工具》File Description中进行配置
 -->
<template>
  <div class="contain">
    <SvgIcon :name="fileIcon" />
    <div v-if="isLoading" class="loading">
      <LoadingImg />
    </div>
    <div class="file-info">
      <span class="file-name">{{ fileInfo.fileName }}</span>
      <span v-if="fileStatus === 'green'" class="file-extension">
        {{ fileInfo.fileExtension.toUpperCase() }}, {{ formatFileSize(fileData.file.size) || 0 }}
      </span>
      <span v-else class="file-extension">
        {{ fileStatusMap.get(fileStatus) }}
      </span>
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
}

const props = defineProps<IProps>();

const { fileData, kbId } = toRefs(props);

const isLoading = computed(() => {
  return fileStatus.value !== 'green';
});

const fileInfo = computed(() => {
  return parseFileName(fileData.value.file_name);
});

const fileIcon = computed(() => {
  if (fileStatus.value !== 'green') {
    return 'file-unknown';
  }
  return 'file-' + iconMap.get(fileInfo.value.fileExtension);
});

const fileStatus = ref(fileData.value.status);

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
  ['red', '上传失败'],
]);

const getDetail = () => {
  let timer = ref(null);
  // if (fileData.value.status === 'green' || fileData.value.status === 'red') {
  //   console.log(fileData.value, 'over');
  //   clearInterval(timer.value);
  //   timer.value = null;
  // } else {
  timer.value = setInterval(async () => {
    if (fileData.value.status === 'green' || fileData.value.status === 'red') {
      console.log(fileData.value, 'over');
      clearInterval(timer.value);
      timer.value = null;
    } else {
      await resultControl(
        await urlResquest.fileList({
          kb_id: kbId.value,
          file_id: fileData.value.file_id,
        })
      );
    }
  }, 10000000);
  // }
};

onMounted(() => {
  getDetail();
  console.log(fileData.value);
  // fileData.value.status = 'gray';
});
</script>

<style lang="scss" scoped>
.contain {
  position: relative;
  width: calc((100% - 16px) / 3);
  height: 62px;
  padding: 8px;
  display: flex;
  align-items: center;
  border-radius: 12px;
  background-color: #fff;
  box-shadow: 0 8px 16px 0 #0000000d;

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

  .file-info {
    margin-left: 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;

    //.file-name {
    //}
    //
    //.file-extension {
    //}
    span {
      min-height: 20px;
      font-family: PingFang SC;
      font-weight: 500;
      font-size: 12px;
      line-height: 20px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
  }
}
</style>
