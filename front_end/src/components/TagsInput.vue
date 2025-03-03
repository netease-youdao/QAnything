<template>
  <div>
    <a-input
      ref="inputRef"
      v-model:value="inputValue"
      type="text"
      size="small"
      placeholder="多个标签用 , 隔开"
      :style="{ width: '178px' }"
      @blur="tagConfirm"
      @keyup.enter="tagConfirm"
    />
  </div>
</template>

<script setup lang="ts">
import message from 'ant-design-vue/es/message';

/**
 * confirm-tag 新增标签调用
 */
const emit = defineEmits(['confirm-tag']);

const inputValue = ref('');
const inputTags = ref<string[]>([]);

const tagConfirm = () => {
  if (inputValue.value.trim().length !== 0) {
    const newTag = inputValue.value.trim();
    handleInputConfirm(newTag);
  }
  inputValue.value = '';
};

// 确认添加标签，焦点，enter 触发
const handleInputConfirm = newTag => {
  // 将新输入的 tag 字符串处理后的 数组
  const tagsArrNew: string[] = newTag.split(',');
  if (newTag.length !== 0) {
    // 检查是否符合规则
    if (!checkTag(newTag)) {
      message.error('标签只可以输入: 汉字 字母 英文逗号 下划线');
    } else if (tagsArrNew.length > 100) {
      message.error('最多添加100个标签');
    } else {
      tagsArrNew.forEach(item => {
        if (item.length > 30) {
          message.error('单个标签长度不能超过30');
          return;
        } else {
          inputTags.value.push(item);
        }
      });
      emit('confirm-tag', inputTags.value);
    }
  }
  inputValue.value = '';
  inputTags.value = [];
};

// 检查添加的标签 是否符合规则 : 汉字 字母 ,(逗号为英文逗号) _
const checkTag = (tag: string) => {
  const regex = /^[\u4e00-\u9fa5a-zA-Z0-9,_]+$/;
  return regex.test(tag);
};
</script>

<style lang="scss" scoped></style>
