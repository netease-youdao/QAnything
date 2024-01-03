<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-10-30 17:47:34
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-26 14:53:17
 * @FilePath: /qanything-open-source/src/views/Home.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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
// import dayjs from 'dayjs';
import DefaultPage from '@/components/Defaultpage.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
// import urlResquest from '@/services/urlConfig';
import Chat from '@/components/Chat.vue';
import OptionList from '@/components/OptionList.vue';
// import { resultControl } from '@/utils/utils';
// import { message } from 'ant-design-vue';

// const { showDefault, currentId } = storeToRefs(useKnowledgeBase());
const { showDefault } = storeToRefs(useKnowledgeBase());

// const { setDefault, setKnowledgeBaseList, setCurrentId, getList } = useKnowledgeBase();
const { setDefault, getList } = useKnowledgeBase();

//需要判断是否有知识库 如果没有知识库 展示default内容
// const showDefault = ref(knowledgeBaseList.value.length ? false : true);

//开始回答后执行的操作
//1.展示聊天界面
//2.默认知识库显示出来
const change = str => {
  setDefault(str);
  getList();
};

//获取知识库列表
// const getList = async () => {
//   try {
//     const res: any = await resultControl(await urlResquest.kbList());
//     if (res && res.length > 0) {
//       setKnowledgeBaseList(res);

//       setDefault(pageStatus.normal);
//       if (currentId.value === '') {
//         setCurrentId(res[0].id);
//       }
//     } else {
//       setDefault(pageStatus.default);
//     }
//   } catch (e) {
//     message.error(e.msg || '请求失败');
//   }
// };

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
