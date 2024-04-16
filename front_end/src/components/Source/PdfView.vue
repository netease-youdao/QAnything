<template>
  <div class="pdf-view-comp">
    <vue-office-pdf :src="sourceUrl" @error="errorHandler" />
  </div>
</template>
<script lang="ts" setup>
import VueOfficePdf from '@vue-office/pdf';

defineProps({
  sourceUrl: {
    type: String,
    required: true,
  },
});

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
.pdf-view-comp {
  width: fit-content;
}
.vue-office-pdf-wrapper {
  width: 720px !important;
}
</style>
