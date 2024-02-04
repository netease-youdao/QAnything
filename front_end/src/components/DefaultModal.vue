<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 17:55:20
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-29 11:06:28
 * @FilePath: /ai-demo/src/components/DefaultModal.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <Teleport to="body">
    <div class="private">
      <a-modal
        v-model:open="showModal"
        :confirm-loading="confirmLoading"
        centered
        width="480px"
        wrap-class-name="private-modal"
        :ok-text="common.confirm"
        :cancel-text="common.cancel"
        @ok="handleOk"
      >
        <template #title>
          <div class="private-title"><SvgIcon class="info" name="info"></SvgIcon>{{ content }}</div>
        </template>
      </a-modal>
    </div>
  </Teleport>
</template>
<script lang="ts" setup>
import { useChat } from '@/store/useChat';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;
const { showModal } = storeToRefs(useChat());
const emits = defineEmits(['ok']);
defineProps({
  content: {
    type: String,
    required: true,
  },
  confirmLoading: {
    type: Boolean,
    required: true,
  },
});

const handleOk = async () => {
  emits('ok');
};
</script>
<style lang="scss" scoped>
.info {
  width: 24px;
  height: 24px;
}

.private-title {
  display: flex;
  align-items: center;
  font-size: 16px;
  font-weight: normal;
  line-height: 24px;
  color: #2e2f33;
  .info {
    margin-right: 12px;
  }
}
</style>

<style lang="scss">
//删除弹框相关修改
.private-modal {
  background: rgba(0, 0, 0, 0.7);
  .ant-modal-content {
    padding: 32px 24px 18px 24px;
    font-size: 16px;
    font-weight: 500;
    color: $title1;
  }
  .ant-modal-title {
    margin-bottom: 8px;
  }
  .ant-modal-body {
    margin-left: 40px;
    font-size: 14px;
    color: $title1;
  }
  .ant-modal-footer {
    margin-top: 58px;
    .ant-btn {
      border-radius: 4px;
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      padding: 5px 20px;
      border-color: #dfe3eb;
      color: #222222;
    }

    .ant-btn-primary {
      background: #5a47e5 !important;
      color: #ffffff;
    }
  }

  .ant-modal-close {
    width: 16px;
    height: 16px;
  }
  .ant-modal-close-x {
    line-height: 16px;
  }
}
</style>
