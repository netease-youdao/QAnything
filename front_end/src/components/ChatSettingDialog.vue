<template>
  <Teleport to="body">
    <a-modal
      v-model:open="showSettingModal"
      :title="'模型设置'"
      centered
      width="480px"
      wrap-class-name="model-set-modal"
      :footer="null"
    >
      <div class="model-set-dialog-comp">
        <!--          <div class="select-model">-->
        <!--            <div class="title">选择模型</div>-->
        <!--            <a-select-->
        <!--              ref="select"-->
        <!--              v-model:value="selectValue"-->
        <!--              class="model-set-select"-->
        <!--              :options="privatizationInfo?.status === 2 ? options : optionsNoPriv"-->
        <!--              :get-popup-container="getContainer"-->
        <!--              @change="handleChange"-->
        <!--            />-->
        <!--            <div class="tokens-points">{{ tokensObj[selectValue] }}</div>-->
        <!--          </div>-->
        <!--          <div class="token-length">-->
        <!--            <div class="title">回复上限</div>-->
        <!--            <div class="slider">-->
        <!--              <a-slider-->
        <!--                v-if="selectValue === 'QAnything 16k'"-->
        <!--                v-model:value="maxToken"-->
        <!--                :min="1024"-->
        <!--                :max="4096"-->
        <!--                :marks="marks16K"-->
        <!--              />-->
        <!--              <a-slider v-else v-model:value="maxToken" :min="512" :max="1024" :marks="marks" />-->
        <!--            </div>-->
        <!--          </div>-->
        <!--          <div class="model-ability">-->
        <!--            <div class="title">模型能力</div>-->
        <!--            <a-checkbox v-model:checked="isNetwork" @change="changeNetwork">联网检索</a-checkbox>-->
        <!--            <div class="hybrid">-->
        <!--              <a-checkbox v-model:checked="isHybrid">混合检索</a-checkbox>-->
        <!--              <a-tooltip>-->
        <!--                <template #title>使用向量检索与全文<br />检索的综合结果返回</template>-->
        <!--                <img src="@/assets/home/tip-icon.png" alt="icon" />-->
        <!--              </a-tooltip>-->
        <!--            </div>-->
        <!--          </div>-->
        <div class="footer">
          <a-button @click="handleCancel">取消</a-button>
          <a-button type="primary" @click="handleOk">确定</a-button>
        </div>
      </div>
    </a-modal>
  </Teleport>
</template>

<script setup lang="ts">
import { useChat } from '@/store/useChat';

const { showSettingModal } = storeToRefs(useChat());

const emit = defineEmits(['confirm']);

// const settingData = reactive({});

const handleOk = () => {
  emit('confirm');
  // 关闭弹窗
  showSettingModal.value = false;
};

const handleCancel = () => {
  // 关闭弹窗
  showSettingModal.value = false;
};
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
    :deep(.ant-select-item-option-selected) {
      background: #eeecfc !important;
      color: #5a47e5 !important;
    }
  }
  :deep(.ant-btn) {
    width: 68px;
    height: 32px;
  }
  :deep(.ant-btn-primary) {
    margin-left: 16px !important;
  }
  :deep(.ant-slider-track) {
    height: 8px;
    background: #8868f1;
    border-radius: 30px;
  }
  :deep(.ant-slider-rail) {
    height: 8px;
    background: #ebeef6;
    border-radius: 30px;
  }
  :deep(.ant-slider-step) {
    display: none;
  }
  :deep(.ant-slider-handle) {
    &::after {
      box-shadow: 0 0 0 2px #8868f1;
      inset-block-start: 1px;
    }
  }
  :deep(.ant-slider-mark-text) {
    margin-top: 4px;
    color: #666666;
    font-size: 14px;
  }
  :deep(.ant-checkbox-checked .ant-checkbox-inner) {
    background-color: #5a47e5;
    border-color: #5a47e5;
  }
  :deep(.ant-checkbox + span) {
    padding-inline-end: 0;
  }
  :deep(
      .ant-checkbox-wrapper:not(.ant-checkbox-wrapper-disabled):hover
        .ant-checkbox-checked:not(.ant-checkbox-disabled)
        .ant-checkbox-inner
    ) {
    background-color: #5a47e5;
    border-color: #5a47e5 !important;
  }
  :deep(
      .ant-checkbox-wrapper:not(.ant-checkbox-wrapper-disabled):hover .ant-checkbox-inner,
      :where(.css-dev-only-do-not-override-3m4nqy).ant-checkbox:not(.ant-checkbox-disabled):hover
        .ant-checkbox-inner
    ) {
    border-color: #5a47e5;
  }
  :deep(
      .ant-checkbox-wrapper:not(.ant-checkbox-wrapper-disabled):hover
        .ant-checkbox-checked:not(.ant-checkbox-disabled):after
    ) {
    border-color: #5a47e5;
  }
}
</style>
