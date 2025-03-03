<template>
  <Teleport to="body">
    <a-modal
      v-model:open="chatSourceVisible"
      centered
      :title="common.dataSource"
      wrap-class-name="chat-source-modal"
      :footer="null"
    >
      <!--      <div class="chat-source">-->
      <!--        <PdfView v-if="sourceType === 'pdf' && sourceUrl" :source-url="sourceUrl" />-->
      <!--        <DocxView v-if="sourceType === 'docx' && sourceUrl" :source-url="sourceUrl" />-->
      <!--        <ExcelView v-if="sourceType === 'xlsx' && sourceUrl" :source-url="sourceUrl" />-->
      <!--        <a-image-->
      <!--          v-if="imageArr.includes(sourceType) && sourceUrl"-->
      <!--          :src="sourceUrl"-->
      <!--          :preview-mask="false"-->
      <!--        />-->
      <!--        <div v-if="sourceType === 'txt'" class="txt" style="white-space: pre-wrap">-->
      <!--          {{ textContent }}-->
      <!--        </div>-->
      <!--        <HighLightMarkDown v-if="sourceType === 'md'" class="txt" :content="textContent" />-->
      <!--      </div>-->
      <Source />
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import Source from './Source/index.vue';
import { useChatSource } from '@/store/useChatSource';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;

// const { chatSourceVisible, sourceUrl, sourceType, textContent } = storeToRefs(useChatSource());
const { chatSourceVisible } = storeToRefs(useChatSource());
// let imageArr = ['jpg', 'png', 'jpeg'];
</script>
<style lang="scss" scoped></style>
