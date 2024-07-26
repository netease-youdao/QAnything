<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:43:45
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-25 18:06:17
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
import { formatFileSize, parseFileName } from '@/utils/utils';
import LoadingImg from '@/components/LoadingImg.vue';

interface IProps {
  fileData: IFileListItem;
}

// eslint-disable-next-line vue/no-setup-props-destructure
const { fileData } = defineProps<IProps>();

const isLoading = computed(() => {
  console.log(fileStatus);
  return fileStatus.value !== 'green';
});

const fileInfo = computed(() => {
  return parseFileName(fileData.file_name);
});

const fileIcon = computed(() => {
  if (fileStatus.value !== 'green') {
    return 'file-unknown';
  }
  return 'file-' + iconMap.get(fileInfo.value.fileExtension);
});

const fileStatus = ref(fileData.status);

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
onMounted(() => {
  console.log(fileData);
});

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
