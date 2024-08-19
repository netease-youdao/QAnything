<template>
  <div
    class="chat-source"
    :style="{
      transform: `scale(${zoomLevel})`,
    }"
  >
    <PdfView v-if="sourceType === 'pdf' && sourceUrl" :source-url="sourceUrl" />
    <DocxView v-if="sourceType === 'docx' && sourceUrl" :source-url="sourceUrl" />
    <ExcelView v-if="sourceType === 'xlsx' && sourceUrl" :source-url="sourceUrl" />
    <a-image
      v-if="imageArr.includes(sourceType) && sourceUrl"
      :src="sourceUrl"
      :preview-mask="false"
    />
    <div
      v-if="sourceType === 'txt' || sourceType === 'csv' || sourceType === 'eml'"
      class="txt"
      style="white-space: pre-wrap"
    >
      {{ textContent }}
    </div>
    <HighLightMarkDown v-if="sourceType === 'md'" class="txt" :content="textContent" />
  </div>
</template>

<script setup lang="ts">
import PdfView from '@/components/Source/PdfView.vue';
import ExcelView from '@/components/Source/ExcelView.vue';
import DocxView from '@/components/Source/DocxView.vue';
import HighLightMarkDown from '@/components/HighLightMarkDown.vue';
import { useChatSource } from '@/store/useChatSource';

const props = defineProps({
  zoomLevel: {
    type: Number,
    require: false,
    default: 1,
  },
});

const { zoomLevel } = toRefs(props);

const { sourceUrl, sourceType, textContent } = storeToRefs(useChatSource());
let imageArr = ['jpg', 'png', 'jpeg'];
</script>

<style lang="scss" scoped>
.chat-source {
  width: 100%;
  min-height: 35vh;
  max-height: calc(90vh - 48px);
  overflow-y: scroll;
  border-radius: 8px;
  display: flex;
  transition: transform 0.3s ease;
  transform-origin: 0 0;

  &::-webkit-scrollbar {
    height: 14px !important;
  }

  .txt {
    width: 680px;
    height: auto;
    padding: 15px 20px 30px 20px;
  }

  :deep(.ant-image) {
    margin: 5px auto;
    max-width: 100%;
  }
}
</style>
