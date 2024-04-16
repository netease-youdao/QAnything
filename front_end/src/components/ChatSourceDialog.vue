<template>
  <Teleport to="body">
    <a-modal
      v-model:open="chatSourceVisible"
      :title="common.dataSource"
      wrap-class-name="chat-source-modal"
      :footer="null"
    >
      <div class="chat-source">
        <PdfView v-if="sourceType === 'pdf' && sourceUrl" :source-url="sourceUrl" />
        <DocxView v-if="sourceType === 'docx' && sourceUrl" :source-url="sourceUrl" />
        <ExcelView v-if="sourceType === 'xlsx' && sourceUrl" :source-url="sourceUrl" />
        <a-image
          v-if="imageArr.includes(sourceType) && sourceUrl"
          :src="sourceUrl"
          :previewMask="false"
        />
        <div v-if="sourceType === 'txt'" class="txt" style="white-space: pre-wrap">
          {{ textContent }}
        </div>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { useChatSource } from '@/store/useChatSource';
import PdfView from './Source/PdfView.vue';
import DocxView from './Source/DocxView.vue';
import ExcelView from './Source/ExcelView.vue';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;

const { chatSourceVisible, sourceUrl, sourceType, textContent } = storeToRefs(useChatSource());

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
  .txt {
    width: 680px;
    padding: 5px 10px;
  }
  :deep(.ant-image) {
    margin: 5px auto;
    max-width: 100%;
  }
}
</style>
