<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-07 17:48:13
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-07 19:44:53
 * @FilePath: \ai-demo\src\components\AddKnowledgeBtn.vue
 * @Description: 
-->
<template>
  <a-button class="add" type="primary" @click="btnClick">
    <template #icon>
      <img class="icon-add" src="../assets/home/icon-add.png" alt="图标" />
    </template>
    新建知识库
  </a-button>
</template>
<script lang="ts" setup>
import { useDebounceFn } from '@vueuse/core';
const emits = defineEmits(['btnClick']);

const btnClick = useDebounceFn(() => {
  emits('btnClick');
}, 300);
</script>
<style lang="scss" scoped>
.add {
  width: 100%;
  height: 40px;

  background-color: $baseColor !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  .icon-add {
    width: 16px;
    height: 16px;
    margin-right: 8px;
  }
}
</style>
