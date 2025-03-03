<template>
  <div class="excel-view-comp">
    <vue-office-excel
      :src="sourceUrl"
      style="height: calc(90vh - 48px); width: 100%"
      @rendered="renderedHandler"
      @error="errorHandler"
    />
  </div>
</template>
<script lang="ts" setup>
import VueOfficeExcel from '@vue-office/excel';
import '@vue-office/excel/lib/index.css';

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
<style lang="scss" scoped>
.excel-view-comp {
  width: 100%;
}
</style>
