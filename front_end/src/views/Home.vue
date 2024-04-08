<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:47:03
 * @FilePath: /QAnything/front_end/src/views/Home.vue
 * @Description: 
-->

<template>
  <div class="page">
    <DefaultPage v-if="showDefault === pageStatus.default" @change="change" />
    <Chat v-else-if="showDefault === pageStatus.normal" />
    <OptionList v-else-if="showDefault === pageStatus.optionlist" />
  </div>
</template>
<script lang="ts" setup>
import { pageStatus } from '@/utils/enum';
import DefaultPage from '@/components/Defaultpage.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import Chat from '@/components/Chat.vue';
import OptionList from '@/components/OptionList.vue';

const { showDefault } = storeToRefs(useKnowledgeBase());

const { setDefault, getList } = useKnowledgeBase();

//开始回答后执行的操作
//1.展示聊天界面
//2.默认知识库显示出来
const change = str => {
  setDefault(str);
  getList();
};

onMounted(() => {
  getList();
});
</script>
<style lang="scss" scoped>
.page {
  width: 100%;
  height: 100%;
  background: #f3f6fd;
}
</style>
