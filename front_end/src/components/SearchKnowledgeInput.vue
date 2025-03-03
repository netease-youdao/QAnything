<template>
  <a-input v-model:value="name" class="search-input" placeholder="搜索知识库名称">
    <template #prefix>
      <img class="search" src="../assets/home/icon-search.png" alt="搜索" />
    </template>
  </a-input>
</template>
<script lang="ts" setup>
import { watchDebounced } from '@vueuse/core';
//新建知识库输入框内容
const name = ref('');
const emits = defineEmits(['search']);

watchDebounced(
  name,
  () => {
    emits('search', name.value);
  },
  { debounce: 500, maxWait: 1000 }
);
</script>
<style lang="scss" scoped>
.search-input {
  width: 100%;
  height: 36px;
  border: 1px solid #ffffff;
  background-color: rgba(255, 255, 255, 0.3);

  :deep(.ant-input) {
    background-color: transparent !important;
  }
  .search {
    width: 16px;
    height: 16px;
  }
}
</style>
