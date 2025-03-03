<template>
  <div class="tags-container">
    <template v-for="tag in tagState.tags.slice(0, 5)" :key="tag">
      <a-tag v-if="tag.tag.length <= 5" :color="tag.color" closable @close="handleClose(tag.tag)">
        {{ tag.tag }}
      </a-tag>
      <a-tooltip v-else color="#fff" placement="topLeft">
        <template #title>
          <span style="color: #666; user-select: text">{{ tag.tag }}</span>
        </template>
        <a-tag :color="tag.color" closable @close="handleClose(tag.tag)">
          {{ `${tag.tag.slice(0, 6)}...` }}
        </a-tag>
      </a-tooltip>
    </template>
    <a-popover v-if="tagState.tags.length > 5" placement="rightTop">
      <template #content>
        <div class="popoverTag">
          <template v-for="tag of tagState.tags.slice(5)" :key="tag">
            <a-tag :color="tag.color" closable @close="handleClose(tag.tag)">
              {{ tag.tag }}
            </a-tag>
          </template>
        </div>
      </template>
      <a-tag color="red" style="cursor: default"> 更多...</a-tag>
    </a-popover>
    <a-popover>
      <template #content>
        支持 英文 汉字 数字 下划线，最多添加100个标签，单个标签长度不能超过30，多个标签以 , 隔开
      </template>
      <a-input
        v-if="tagState.isInput"
        ref="inputRef"
        v-model:value="tagState.inputValue"
        type="text"
        size="small"
        :style="{ width: '78px' }"
        @blur="tagConfirm"
        @keyup.enter="tagConfirm"
      />
      <a-tag v-else style="background: #fff; border-style: dashed" @click="showInput">
        <plus-outlined />
      </a-tag>
    </a-popover>
  </div>
</template>

<script setup lang="ts">
import { PlusOutlined } from '@ant-design/icons-vue';
import message from 'ant-design-vue/es/message';

interface IProps {
  tags: string[];
}

const props = defineProps<IProps>();

type TagsType = {
  tag: string;
  color: string;
};

/**
 * update:tags 更新标签调用(清除、新增)
 * confirm-tag 新增标签调用
 */
const emit = defineEmits(['update:tags', 'confirm-tag']);

// 动态增添标签
interface ITagState {
  tags: TagsType[];
  isInput: boolean;
  inputValue: string;
}

const inputRef = ref();
const tagsColor = ['pink', 'orange', 'green', 'cyan', 'blue', 'purple'];
const tagState = ref<ITagState>({
  tags: props.tags.map((tag, index) => ({
    tag,
    color: computedColor(index),
  })),
  isInput: false,
  inputValue: '',
});

function computedColor(index: number) {
  return tagsColor[index % tagsColor.length];
}

const handleClose = (tag: string) => {
  const newTags = tagState.value.tags.map(item => item.tag).filter(item => item !== tag);
  emit('update:tags', newTags);
  emit('confirm-tag', newTags);
};

// 同步 props.tags变化
watchEffect(() => {
  tagState.value.tags = props.tags.map((tag, index) => ({
    tag,
    color: computedColor(index),
  }));
});

const showInput = () => {
  tagState.value.isInput = true;
  nextTick(() => {
    inputRef.value.focus();
  });
};

const tagConfirm = () => {
  if (tagState.value.inputValue.trim().length !== 0) {
    const newTag = tagState.value.inputValue.trim();
    handleInputConfirm(newTag);
  }
  tagState.value.inputValue = '';
  tagState.value.isInput = false;
};

// 确认添加标签，焦点，enter 触发
const handleInputConfirm = newTag => {
  // 将新输入的 tag 字符串处理后的 数组
  const tagsArrNew: string[] = newTag.split(',');
  // 原来的 tag 数组, 引用
  const tagsArrInitial = tagState.value.tags.map(item => item.tag);
  if (newTag.length !== 0) {
    // 检查是否符合规则
    if (!checkTag(newTag)) {
      message.error('标签只可以输入: 汉字 字母 英文逗号 下划线');
    } else if (tagsArrNew.length + tagsArrInitial.length > 100) {
      message.error('最多添加100个标签');
    } else {
      tagsArrNew.forEach(item => {
        if (item.length > 30) {
          message.error('单个标签长度不能超过30');
          return;
        } else if (item && tagsArrInitial.indexOf(item) === -1) {
          tagsArrInitial.push(item);
        }
      });
      emit('update:tags', tagsArrInitial);
      emit('confirm-tag', tagsArrInitial);
    }
  }
  Object.assign(tagState.value, {
    isInput: false,
    inputValue: '',
  });
};

// 检查添加的标签 是否符合规则 : 汉字 字母 ,(逗号为英文逗号) _
const checkTag = (tag: string) => {
  const regex = /^[\u4e00-\u9fa5a-zA-Z0-9,_]+$/;
  return regex.test(tag);
};
</script>

<style lang="scss" scoped>
.tags-container {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.popoverTag {
  max-width: 330px;
}

.tags-tooltip {
  :deep(.ant-tooltip-inner) {
    cursor: text;
    background-color: #fff;
  }
}
</style>
