<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:43:45
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-24 17:04:20
 * @FilePath: front_end/src/views/QuickStart/children/FileBlock.vue
 * @Description: 这是默认设置,可以在设置》工具》File Description中进行配置
 -->
<template>
  <div class="contain">
    <SvgIcon :name="fileIcon" />
    <div class="file-info">
      <span class="file-name">{{ fileInfo.fileName }}</span>
      <span class="file-extension">
        {{ fileInfo.fileExtension.toUpperCase() }}, {{ formatFileSize(fileData.file.size) || 0 }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { IFileListItem } from '@/utils/types';
import SvgIcon from '@/components/SvgIcon.vue';
import { formatFileSize, parseFileName } from '@/utils/utils';

interface IProps {
  fileData: IFileListItem;
}
// eslint-disable-next-line vue/no-setup-props-destructure
const { fileData } = defineProps<IProps>();

const fileInfo = computed(() => {
  return parseFileName(fileData.file_name);
});

const fileIcon = computed(() => {
  return 'file-' + iconMap.get(fileInfo.value.fileExtension);
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
onMounted(() => {
  console.log(fileData);
});
</script>

<style lang="scss" scoped>
.contain {
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
