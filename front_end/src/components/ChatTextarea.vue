<!--
 * 用 div 的 contenteditable 模拟 textarea
 -->
<template>
  <div class="chat-textarea-container">
    <div class="chat-textarea-wrapper">
      <div
        ref="textareaRef"
        class="chat-textarea"
        contenteditable="plaintext-only"
        @input="inputHandle"
        @keydown="textKeydownHandle($event)"
      />
      <div class="placeholder" @click.prevent="handlePlaceholderClick">
        <span v-if="!inputValue.length">{{ common.problemPlaceholder }}</span>
      </div>
    </div>
    <div class="send-action">
      <slot></slot>
    </div>
  </div>
  <Mention
    ref="mentionRef"
    v-model:active-index="mentionActiveIndex"
    :options="options"
    :position="mentionPosition"
    @selectMention="selectHandle"
  />
</template>

<script setup lang="ts">
import { getLanguage } from '@/language';
import Mention from '@/components/Mention.vue';

const textareaRef = ref<InstanceType<typeof HTMLDivElement>>(null);

const common = getLanguage().common;

interface IProps {
  inputValue: string;
  options: string[];
}

const props = defineProps<IProps>();

// 只读
const { inputValue, options: optionsProps } = toRefs(props);

const emit = defineEmits(['update:inputValue', 'send']);

const cursorPosition = ref(0);

// 输入触发
const inputHandle = (e: Event) => {
  emit('update:inputValue', (e.target as HTMLDivElement).textContent);
  nextTick(() => {
    // 初始化选中条目
    mentionActiveIndex.value = 0;
    // 初始化 mention 内部选中条目的位置
    mentionRef.value.initScroll();
    // 计算新的光标位置
    cursorPosition.value = computedCursorPosition();
    const pointerPosition = cursorPosition.value;
    // 获取内容
    const text = inputValue.value;
    // 从光标位置向前找到第一个 @ 的位置
    const atSymbolIndex = text.lastIndexOf('@', pointerPosition - 1);
    if (atSymbolIndex !== -1) {
      // 获取 @ 之后到 光标 的所有字符
      const atText = text.substring(atSymbolIndex + 1, pointerPosition);
      // 将 atText 在 option 中的项过滤出来
      const filterOption = options.value.filter(item => item.label.includes(atText));
      if (filterOption.length !== 0) {
        options.value = filterOption;
        // 计算坐标 并 直接赋值
        const { left, top } = getPosition();
        mentionPosition.value.left = left + 5;
        mentionPosition.value.top = top + 20;
      } else {
        // 不在 option 中
        initMention();
      }
    } else {
      // 不是 @
      initMention();
    }
  });
};

onMounted(() => {
  nextTick(() => {
    // 获取焦点
    textareaRef.value?.focus();
  });
});

// 阻止placeholder事件冒泡
const handlePlaceholderClick = (e: MouseEvent) => {
  e.stopPropagation();
  textareaRef.value?.focus();
};

// 计算光标索引
const computedCursorPosition = () => {
  const selection = window.getSelection();
  const range = selection.getRangeAt(0);
  // 克隆一个 Range 对象，用于截取文本
  const cloneRange = range.cloneRange();
  const textNode = textareaRef.value?.firstChild;
  if (!textNode) return 0;
  // 选中所有文本
  cloneRange.selectNodeContents(textNode);
  // 设置 range终点位置，也就是模拟选中 光标以前 的节点
  cloneRange.setEnd(range.endContainer, range.endOffset);
  // 计算 选中的长度 -> 光标位置索引
  return cloneRange.toString().length;
};

// 根据索引设置光标位置，index是索引
const resetCursorPosition = (index: number) => {
  if (textareaRef.value) {
    const textNode = textareaRef.value?.firstChild;
    const selection = window.getSelection();
    const range = document.createRange();
    if (!textNode) return;
    try {
      range.setStart(textNode, index);
      range.setEnd(textNode, index);
      // 光标开始和光标结束重叠
      range.collapse(true);
      // 清除选定对象的所有光标对象
      selection.removeAllRanges();
      // 插入新的光标对象
      selection.addRange(range);
    } catch (e) {
      console.error('catch', e);
      range.setStart(textNode, cursorPosition.value);
      range.setEnd(textNode, cursorPosition.value);
      // 光标开始和光标结束重叠
      range.collapse(true);
      // 清除选定对象的所有光标对象
      selection.removeAllRanges();
      // 插入新的光标对象
      selection.addRange(range);
    }
  }
};

// 计算光标坐标
const getPosition = () => {
  // nextTick(() => {
  if (!textareaRef.value) return;
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  const rect = range.getBoundingClientRect();
  return {
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY,
  };
  // });
};

// 1. 聊天框keydown 换行，不允许enter换行，alt/ctrl/shift/meta(Command或win) + enter可换行
// 2. 左右，检测 @ 包含范围
const textKeydownHandle = e => {
  // 首先检查是否按下 Enter 键
  if (e.keyCode === 13) {
    if (e.altKey || e.ctrlKey || e.shiftKey || e.metaKey) {
      // 等待 inputValue.value 更新
      nextTick(() => {
        cursorPosition.value = computedCursorPosition();
        const newValue = inputValue.value.replaceAll(/<br>/g, '\n');
        textareaRef.value.textContent = newValue;
        resetCursorPosition(cursorPosition.value);
      });
    } else {
      e.preventDefault();
      // 判断是不是 Mention 的回车
      mentionPosition.value.top === -9999 && send();
    }
  } else if (e.keyCode === 37 || e.keyCode === 39) {
    // 左右
    inputHandle(e);
  } else if (e.keyCode === 38 || e.keyCode === 40) {
    // 上下
    if (mentionPosition.value.top !== -9999) {
      e.preventDefault();
    }
  }
};

/* -------------*/
// Mention 的引用
const mentionRef = ref<InstanceType<typeof Mention>>(null);

// Mention 的选项
const options = ref([]);

// Mention 的坐标
const mentionPosition = ref({
  left: 0,
  top: -9999,
});

// Mention 选中的 index
const mentionActiveIndex = ref(0);
const initMention = () => {
  mentionPosition.value.top = -9999;
  options.value = [
    ...optionsProps.value.map(item => ({
      label: item,
      value: item,
    })),
  ];
  mentionActiveIndex.value = 0;
};

// Mention 点击选项的回调
const selectHandle = (value: string) => {
  initMention();
  // 点击不需要计算光标位置，因为如果计算那就是计算 Mention 的select
  const pointerPosition = cursorPosition.value;
  // 获取当前 输入 的所有内容
  const currentText = inputValue.value;
  // 从光标位置向前找到第一个 @ 的位置
  const atSymbolIndex = currentText.lastIndexOf('@', pointerPosition - 1);
  // 设置新的文本内容 @符之前的(包括@) + 当前值 + 空格 + 当前光标之后的
  const newValue =
    currentText.substring(0, atSymbolIndex + 1) +
    value +
    ' ' +
    currentText.substring(pointerPosition);
  emit('update:inputValue', newValue);
  textareaRef.value.textContent = newValue;
  resetCursorPosition(pointerPosition + value.length + 1);
};

window.addEventListener('click', () => {
  if (mentionPosition.value.top !== -9999) {
    initMention();
  }
});

// 对 optionsProps 的处理
watchEffect(() => {
  options.value = [
    ...optionsProps.value.map(item => ({
      label: item,
      value: item,
    })),
  ];
});

/* ----------- */
// 发送问答消息
const send = () => {
  emit('send');
  nextTick(() => {
    textareaRef.value?.focus();
  });
};

watch(
  () => props.inputValue,
  () => {
    console.log('props', inputValue.value);
    if (props.inputValue.length === 0) {
      textareaRef.value.textContent = props.inputValue;
    }
  }
);
</script>

<style lang="scss" scoped>
.chat-textarea-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
  background-color: #fff;
  border: 1px solid #d9d9d9;
  border-radius: 18px;

  &:hover {
    border-color: $baseColor;
    transition: border-color 0.3s, height 0s;
  }

  &:not(:hover) {
    border-color: #d9d9d9;
    transition: border-color 0.3s;
  }

  &:focus {
    box-shadow: 0 0 0 2px rgba(5, 145, 255, 0.1);
  }
}

.chat-textarea-wrapper {
  position: relative;
  width: 100%;
  min-height: 47px;
  max-height: 222px;
  overflow-y: auto;

  .chat-textarea {
    width: 100%;
    min-height: 47px;
    line-height: 25px;
    padding: 11px 15px;
    font-size: 14px;
    border-radius: 18px;
    outline: none;
    z-index: 105;
    box-sizing: border-box;
  }

  .placeholder {
    position: absolute;
    padding: 11px 15px 0;
    top: 0;
    left: 0;
    //top: 10px;
    //left: 15px;
    font-size: 14px;
    color: #ccc;
    z-index: 1;
    user-select: none;
    cursor: text;

    span {
      font-size: 14px;
      white-space: pre-wrap;
    }
  }
}

.send-action {
  width: 100%;
  height: 40px;
  padding-right: 10px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  color: #000;
  z-index: 101;

  :deep(.ant-btn-primary) {
    width: 36px;
    height: 26px;
    padding: 8px 10px 8px 8px;
    border-radius: 18px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(300deg, #7b5ef2 1%, #c383fe 97%);
  }

  :deep(.ant-btn-primary:disabled) {
    //height: 36px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(300deg, #7b5ef2 1%, #c383fe 97%);
    color: #fff !important;
    border-color: transparent !important;
  }

  svg {
    width: 24px;
    height: 24px;
  }
}
</style>
