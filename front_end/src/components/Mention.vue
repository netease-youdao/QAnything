<template>
  <div class="mention-container" :style="{ top: position.top + 'px', left: position.left + 'px' }">
    <ul ref="mentionList" class="mention-list">
      <li
        v-for="(option, index) of options"
        :key="option.value"
        :class="['mention-item', index === activeIndex ? 'mention-item-active' : '']"
        @click="selectHandle(option.value)"
      >
        <span class="mention-item-text">
          {{ option.label }}
        </span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
type OptionsType = {
  label: string;
  value: string;
};

type PositionType = {
  left: number;
  top: number;
};

interface IProps {
  activeIndex: number;
  options: OptionsType[];
  position: PositionType;
}

const props = defineProps<IProps>();
const emit = defineEmits(['selectMention', 'update:activeIndex']);

const { activeIndex, options, position } = toRefs(props);

const selectHandle = (value: any) => {
  emit('selectMention', value);
};

const keyHandle = (type: 'down' | 'up') => {
  let index = activeIndex.value;
  if (type === 'down') {
    index++;
    if (index >= options.value.length - 1) {
      index = 0;
    }
  } else if (type === 'up') {
    index--;
    if (index < 0) {
      index = options.value.length - 1;
    }
  }
  emit('update:activeIndex', index);
  scrollToActiveItem();
};

const mentionList = ref(null);
const scrollToActiveItem = () => {
  nextTick(() => {
    const list = mentionList.value;
    const activeItem = list.children[activeIndex.value];
    if (list && activeItem) {
      const listHeight = list.clientHeight;
      const itemHeight = activeItem.clientHeight;
      const itemTop = activeItem.offsetTop;
      const scrollTop = list.scrollTop;

      if (itemTop < scrollTop) {
        list.scrollTop = itemTop;
      } else if (itemTop + itemHeight > scrollTop + listHeight) {
        list.scrollTop = itemTop + itemHeight - listHeight;
      }
    }
  });
};
const initScroll = () => {
  nextTick(() => {
    const list = mentionList.value;
    if (list) {
      list.scrollTop = 0;
    }
  });
};

defineExpose({ initScroll });

window.addEventListener('keydown', e => {
  if (position.value.top === -9999) return;
  if (e.key === 'ArrowDown') {
    keyHandle('down');
  } else if (e.key === 'ArrowUp') {
    keyHandle('up');
  } else if (e.key === 'Enter') {
    selectHandle(options.value[activeIndex.value].value);
  }
});

watchEffect(() => {
  console.log(position.value, 'aaa');
});
</script>

<style lang="scss" scoped>
.mention-container {
  position: fixed;
  display: flex;
  flex-direction: column;
  //top: -9999px;
  //left: calc(509px - 280px);
  top: 0;
  z-index: 199;
  background-color: #fff;
  line-height: 1.5;
  border-radius: 8px;
  outline: none;
  box-shadow: 0 6px 16px 0 rgba(0, 0, 0, 0.08), 0 3px 6px -4px rgba(0, 0, 0, 0.12),
    0 9px 28px 8px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.mention-list {
  max-height: 85px;
  margin-bottom: 0;
  overflow-y: auto;

  .mention-item {
    position: relative;
    min-width: 100px;
    padding: 5px 12px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    display: block;
    cursor: pointer;
    transition: background 0.3s ease;
    color: rgba(0, 0, 0, 0.88);

    &:hover {
      background-color: rgba(0, 0, 0, 0.06);
      //background-color: #e5e7ed;
    }
    .mention-item-text {
      font-size: 14px;
    }
  }

  &::-webkit-scrollbar {
    width: 0;
  }

  .mention-item-active {
    //background-color: rgba(0, 0, 0, 0.06);
    background-color: #e5e7ed !important;
  }
}
</style>
