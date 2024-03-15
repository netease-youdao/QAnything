<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-26 11:43:52
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-02 11:09:45
 * @FilePath: /qanything-open-source/src/components/AddInput.vue
 * @Description: 
-->
<template>
  <a-input v-model:value="kb_name" class="add-input" :placeholder="common.newPlaceholder">
    <template #suffix>
      <div class="add-button" @click="addKb">{{ common.new }}</div>
    </template>
  </a-input>
</template>
<script lang="ts" setup>
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
const { getList, setCurrentId, setCurrentKbName, setDefault } = useKnowledgeBase();
const { selectList } = storeToRefs(useKnowledgeBase());
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { pageStatus } from '@/utils/enum';
const { setModalVisible } = useKnowledgeModal();
const { modalVisible } = storeToRefs(useKnowledgeModal());
const kb_name = ref('');
import { getLanguage } from '@/language/index';

const common = getLanguage().common;

// const emits = defineEmits(['add']);

const addKb = async () => {
  if (!kb_name.value.length) {
    message.error(common.errorKnowledge);
    return;
  }

  try {
    const res: any = await resultControl(await urlResquest.createKb({ kb_name: kb_name.value }));
    kb_name.value = '';
    console.log(res);
    setCurrentId(res?.kb_id);
    setCurrentKbName(res?.kb_name);
    selectList.value.push(res?.kb_id);
    await getList();
    setModalVisible(!modalVisible.value);
    setDefault(pageStatus.optionlist);
  } catch (e) {
    console.log(e);
    message.error(e.msg || common.error);
  }
};
</script>

<style lang="scss" scoped>
.add-button {
  cursor: pointer;
  width: 52px;
  height: 32px;
  border-radius: 4px;
  background: #5a47e5;
}
.add-button {
  font-size: 14px;
  font-weight: 500;
  line-height: 32px;
  text-align: center;
  color: #fff;
}
</style>
