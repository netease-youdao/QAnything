<template>
  <div class="docx-view-comp">
    <vue-office-docx :src="sourceUrl" @rendered="renderedHandler" @error="errorHandler" />
  </div>
</template>
<script lang="ts" setup>
import VueOfficeDocx from '@vue-office/docx';
import '@vue-office/docx/lib/index.css';

defineProps({
  sourceUrl: {
    type: String,
    required: true,
  },
});

function renderedHandler() {
  console.log('渲染完成');
}

function errorHandler(e) {
  console.log('渲染失败', e);
}
</script>
<style lang="scss">
// .vue-office-docx .docx-wrapper > section.docx {
//   width: auto !important;
// }
// .docx-wrapper {
//   padding: 0 !important;
// }
</style>
