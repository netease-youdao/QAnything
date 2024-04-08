<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-11 14:45:05
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-11 16:13:21
 * @FilePath: /ai-demo/src/components/AddSiderCard.vue
 * @Description: 
-->

<template>
  <div class="card active" :style="props.style">
    <div class="title">
      <div class="editing">
        <p class="title-text">
          <a-input v-model:value="title" type="text" placeholder="请输入知识库名称"></a-input>
        </p>
        <span class="icon-box">
          <SvgIcon class="edit" name="card-confirm" @click="ok"></SvgIcon>
          <SvgIcon class="delete" name="card-cancel" @click="close"></SvgIcon>
        </span>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
const title = ref('');
const emit = defineEmits(['create', 'close']);
const props = defineProps({
  style: {
    type: Object,
    default: () => {
      return {
        width: 'calc(100% - 40px)',
      };
    },
  },
});

//确定修改
const ok = async () => {
  emit('create', title.value);
  title.value = '';
};
//取消修改
const close = () => {
  title.value = '';
  emit('close');
};
</script>
<style lang="scss" scoped>
.card {
  overflow: hidden;
  position: relative;
  height: 72px;
  margin: 0 20px;
  margin-bottom: 16px;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
  border: 1px solid transparent;
  user-select: none;

  .title {
    display: flex;
    align-items: center;
    color: $title1;
    font-size: 14px;
    height: 22px;
    line-height: 22px;
    margin: 14px 1px 0px 16px;

    .normal {
      .title-text {
        width: 169px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }
    }

    .editing {
      margin-right: 8px;

      .title-text {
        width: 168px;
        height: 28px;

        :deep(.ant-input) {
          padding: 2px 11px;
        }
      }

      .icon-box {
        right: 8px;
      }

      :deep(.ant-input) {
        background: #f3f3f3;
        border-color: #f3f3f3;
      }
    }

    .edit,
    .delete {
      width: 16px;
      height: 16px;
      cursor: pointer;
    }

    .edit {
      margin-right: 8px;
    }

    .icon-box {
      position: absolute;
      right: 14px;
      top: 17px;
    }
  }

  .time {
    margin-top: 2px;
    margin-left: 16px;
    color: $title3;
    font-size: 12px;
    height: 20px;
    line-height: 20px;
  }
}
.active {
  border: 2px solid #4d71ff;
}

.fade-enter-active {
  transition: all 0.3s ease-out;
}

.fade-leave-active {
  opacity: 0.5;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
