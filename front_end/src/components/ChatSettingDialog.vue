<template>
  <Teleport to="body">
    <a-modal
      v-model:open="showSettingModal"
      :title="common.modelSettingTitle"
      centered
      width="700px"
      wrap-class-name="model-set-modal"
      :footer="null"
      :force-render="true"
    >
      <div class="model-set-dialog-comp">
        <div class="select-model">
          <ChatSettingForm ref="chatSettingFormRef" />
        </div>
        <div class="footer">
          <a-button @click="handleCancel">{{ common.cancel }}</a-button>
          <a-button type="primary" style="width: auto; margin-left: 10px" @click="handleConfirm">
            {{ common.confirmApplication }}
          </a-button>
        </div>
      </div>
    </a-modal>
  </Teleport>
</template>

<script setup lang="ts">
import { message } from 'ant-design-vue';
import { useChat } from '@/store/useChat';
import { useChatSetting } from '@/store/useChatSetting';
import ChatSettingForm from '@/components/ChatSettingForm.vue';
import { getLanguage } from '@/language';

const { showSettingModal } = storeToRefs(useChat());
const { setChatSettingConfigured } = useChatSetting();

const chatSettingFormRef = ref<InstanceType<typeof ChatSettingForm>>();
const { common } = getLanguage();

const handleCancel = () => {
  showSettingModal.value = false;
};

const handleOk = async (fn?: Function) => {
  const checkRes = await chatSettingFormRef.value.onCheck();
  if (!Object.hasOwn(checkRes, 'errorFields')) {
    fn && fn(checkRes);
    return true;
  }
  return false;
};

const handleOkCB = checkRes => {
  showSettingModal.value = false;
  setChatSettingConfigured(checkRes);
  message.success('应用成功');
};

const handleConfirm = () => {
  handleOk(handleOkCB);
};

defineExpose({ handleOk });
</script>

<style lang="scss" scoped>
.model-set-dialog-comp {
  width: 100%;
  height: 100%;
  font-family: PingFang SC;
  color: #222222;

  .title {
    font-size: 14px;
    font-weight: 500;
    line-height: 22px;
    margin-right: 20px;
  }

  .select-model {
    margin-bottom: 24px;

    .tokens-points {
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      color: #2e2f33;
      margin-left: 12px;
    }
  }

  .select-model,
  .model-ability,
  .token-length {
    width: 100%;
    display: flex;
    align-items: center;

    .slider {
      flex: 1;
    }
  }

  .token-length {
    align-items: start;
    margin-bottom: 24px;

    .title {
      margin-top: 7px;
    }
  }

  .model-ability {
    .hybrid {
      display: flex;
      align-items: center;
      margin-left: 34px;

      img {
        width: 16px;
        height: 16px;
        margin-left: 4px;
      }
    }
  }

  .footer {
    margin-top: 48px;
    display: flex;
    justify-content: end;
  }

  .model-set-select {
    margin-left: 16px;

    :deep(.ant-select-selector) {
      width: 156px;
      height: 40px;
      display: flex;
      align-items: center;
    }
  }
}
</style>
