<template>
  <a-pagination
    v-model:current="current1"
    v-model:page-size="pageSize1"
    :total="knowledgeBaseList.length"
    :show-total="total => `共 ${total} 条`"
    @change="myChange"
  />
</template>
<script lang="ts" setup>
const current1 = ref<number>(1);
const pageSize1 = ref<number>(20);
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
const { knowledgeBaseList } = storeToRefs(useKnowledgeBase());
const myChange = (page, pageSize) => {
  console.log('page:  ', page);
  console.log('pageSize:  ', pageSize);
};
</script>
<style lang="scss" scoped></style>
