<template>
  <a-form
    ref="formRef"
    :model="chatSettingForm"
    :rules="rules"
    :label-col="{ span: 6 }"
    :wrapper-col="{ span: 24 }"
    style="width: 100%"
    label-align="left"
    @finish="onSubmit"
  >
    <a-form-item :label="common.modelProviderLabel" name="modelType">
      <a-select
        v-model:value="chatSettingForm.modelType"
        :placeholder="common.selectModel"
        @select="selectChange"
      >
        <a-select-option value="openAI">openAI</a-select-option>
        <a-select-option value="ollama">ollama</a-select-option>
        <a-select-option
          v-for="item of chatSettingConfigured.filter(
            i => i.modelType !== 'openAI' && i.modelType !== 'ollama'
          )"
          :key="item.customId"
          :value="item.customId"
        >
          {{ item.modelName.length === 0 ? common.customModelType : item.modelName }}
        </a-select-option>
      </a-select>
    </a-form-item>
    <a-form-item
      v-if="chatSettingForm.modelType === '自定义模型配置'"
      ref="modelName"
      :label="common.modelNameLabel"
      name="modelName"
    >
      <a-input v-model:value="chatSettingForm.modelName" aria-autocomplete="none" />
    </a-form-item>
    <a-form-item
      v-if="chatSettingForm.modelType !== 'ollama'"
      ref="apiKey"
      :label="common.apiKeyLabel"
      name="apiKey"
    >
      <a-input-password
        v-model:value="chatSettingForm.apiKey"
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
      />
    </a-form-item>
    <a-form-item ref="apiBase" :label="common.apiPathLabel" name="apiBase">
      <a-input v-model:value="chatSettingForm.apiBase" aria-autocomplete="none" />
    </a-form-item>
    <a-form-item ref="apiModelName" :label="common.apiModelNameLabel" name="apiModelName">
      <a-input v-model:value="chatSettingForm.apiModelName" aria-autocomplete="none" />
    </a-form-item>
    <div class="form-item-inline">
      <a-form-item
        ref="apiContextLength"
        :label="common.apiContextLengthLabel"
        name="apiContextLength"
      >
        <a-slider
          v-model:value="chatSettingForm.apiContextLength"
          :min="4096"
          :max="8192"
          :step="1"
        />
      </a-form-item>
      <a-form-item name="apiContextLength">
        <a-input-number
          v-model:value="chatSettingForm.apiContextLength"
          :min="4096"
          :step="1"
          style="margin-left: 16px"
          :precision="0"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item ref="maxToken" :label="common.maxTokenLabel" name="maxToken">
        <a-slider
          v-model:value="chatSettingForm.maxToken"
          :min="1"
          :max="chatSettingForm.apiContextLength / TOKENRATIO"
          :step="1"
        />
      </a-form-item>
      <a-form-item name="maxToken">
        <a-input-number
          v-model:value="chatSettingForm.maxToken"
          :min="1"
          :max="chatSettingForm.apiContextLength / TOKENRATIO"
          :step="1"
          style="margin-left: 16px"
          :precision="0"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item
        ref="context"
        :label="`${common.contextLabel}（${contextLength}）`"
        name="context"
      >
        <a-slider v-model:value="chatSettingForm.context" :min="0" :max="contextLength" :step="1" />
      </a-form-item>
      <a-form-item name="context">
        <a-input-number
          v-model:value="chatSettingForm.context"
          :min="0"
          :max="contextLength"
          :step="1"
          style="margin-left: 16px"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item ref="temperature" label="Temperature" name="temperature">
        <a-slider v-model:value="chatSettingForm.temperature" :min="0" :max="1" :step="0.01" />
      </a-form-item>
      <a-form-item name="temperature">
        <a-input-number
          v-model:value="chatSettingForm.temperature"
          :min="0"
          :max="1"
          :step="0.01"
          style="margin-left: 16px"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item ref="top_P" label="top_P" name="top_P">
        <a-slider v-model:value="chatSettingForm.top_P" :min="0" :max="1" :step="0.01" />
      </a-form-item>
      <a-form-item name="top_P">
        <a-input-number
          v-model:value="chatSettingForm.top_P"
          :min="0"
          :max="1"
          :step="0.01"
          style="margin-left: 16px"
        />
      </a-form-item>
    </div>
    <a-form-item :label="common.capabilitiesLabel" name="capabilities">
      <a-checkbox-group v-model:value="capabilitiesOptionsState" style="width: 100%">
        <a-row>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.networkSearchDescription }}</p>
              </template>
              <a-checkbox value="onlineSearch">{{ common.networkSearch }}</a-checkbox>
            </a-popover>
          </a-col>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.mixedSearchDescription }}</p>
              </template>
              <a-checkbox value="mixedSearch">{{ common.mixedSearch }}</a-checkbox>
            </a-popover>
          </a-col>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.onlySearchDescription }}</p>
              </template>
              <a-checkbox value="onlySearch">{{ common.onlySearch }}</a-checkbox>
            </a-popover>
          </a-col>
        </a-row>
      </a-checkbox-group>
    </a-form-item>
    <a-form-item v-if="chatSettingForm.modelType === '自定义模型配置'" :wrapper-col="{ span: 24 }">
      <a-button type="primary" html-type="submit" style="margin-left: 0 !important; width: auto">
        {{ common.saveModel }}
      </a-button>
    </a-form-item>
  </a-form>
</template>

<script setup lang="ts">
import { IChatSetting } from '@/utils/types';
import type { Rule } from 'ant-design-vue/es/form';
import { useChatSetting } from '@/store/useChatSetting';
import { getLanguage } from '@/language';
import { message } from 'ant-design-vue';

defineExpose({ onCheck });

const common = getLanguage().common;

interface IProps {
  contextLength: number;
}

const props = defineProps<IProps>();
const contextLength = toRef(props, 'contextLength');

const { chatSettingConfigured } = storeToRefs(useChatSetting());
const { setChatSettingConfigured } = useChatSetting();

const formRef = ref(null);
const capabilitiesOptionsState = ref([]);

const chatSettingForm = ref<IChatSetting>();

// maxToken是上下文token的 1/TOKENRATIO 倍
const TOKENRATIO = 2;

const rules: Record<string, Rule[]> = {
  modelType: [
    { required: true, message: `Please Select ${common.modelProviderLabel}`, trigger: 'change' },
  ],
  modelName: [
    { required: true, message: `Please Input ${common.modelNameLabel}`, trigger: 'change' },
  ],
  apiKey: [{ required: true, message: `Please Input ${common.apiKeyLabel}`, trigger: 'change' }],
  apiBase: [{ required: true, message: `Please Input ${common.apiPathLabel}`, trigger: 'change' }],
  apiModelName: [
    { required: true, message: `Please Input ${common.apiModelNameLabel}`, trigger: 'change' },
  ],
  apiContextLength: [
    { required: true, message: `Please Input ${common.apiContextLengthLabel}`, trigger: 'change' },
  ],
  maxToken: [
    { required: true, message: `Please Input ${common.maxTokenLabel}`, trigger: 'change' },
  ],
};

// 主动检测是否通过
async function onCheck() {
  try {
    await formRef.value.validateFields();
    return chatSettingForm.value;
  } catch (errorInfo) {
    return errorInfo;
  }
}

// 如果验证通过，则调用submit
function onSubmit() {
  setChatSettingConfigured(chatSettingForm.value);
  message.success('添加成功');
}

// 选择触发的函数
const selectChange = (value: 'openAI' | 'ollama' | number) => {
  console.log('select', value);
  // 重新设置当前的表单项
  if (value === 'openAI') {
    chatSettingForm.value = JSON.parse(JSON.stringify(chatSettingConfigured.value[0]));
  } else if (value === 'ollama') {
    chatSettingForm.value = JSON.parse(JSON.stringify(chatSettingConfigured.value[1]));
  } else {
    const chatForm = chatSettingConfigured.value.find(item => item.customId === value);
    // find出来是浅拷贝
    chatSettingForm.value = JSON.parse(JSON.stringify(chatForm));
  }
};

// 转化checkbox-group和自己定义的表单项，0为表单项 -> checkbox-group（初始化用）
function transformCheckbox(type: 0 | 1) {
  if (type === 0) {
    capabilitiesOptionsState.value = [];
    const capabilities = chatSettingForm.value.capabilities;
    for (let item in capabilities) {
      if (capabilities[item]) {
        capabilitiesOptionsState.value.push(item);
      }
    }
  } else {
    for (let item in chatSettingForm.value.capabilities) {
      chatSettingForm.value.capabilities[item] = false;
    }
    capabilitiesOptionsState.value.forEach(item => {
      chatSettingForm.value.capabilities[item] = true;
    });
  }
}

// 初始化表单项，将active = true的表单项作为默认
const initForm = () => {
  const activeForm = chatSettingConfigured.value.find(item => item.active === true);
  chatSettingForm.value = JSON.parse(JSON.stringify(activeForm));
};

// 监听多选框，转化到form表单项中
watch(
  () => capabilitiesOptionsState.value,
  () => transformCheckbox(1)
);

// 监听表单项，转化表单项 -> checkbox-group
watch(
  () => chatSettingForm.value,
  () => {
    transformCheckbox(0);
    console.log('当前的表单为：', chatSettingForm.value);
  }
);

// 监听上下文token数，上下文token一变，回复最大token自动变为最大（上下文token / TOKENRATIO）
watch(
  () => chatSettingForm.value?.apiContextLength,
  () => {
    chatSettingForm.value.maxToken = chatSettingForm.value.apiContextLength / TOKENRATIO;
  }
);

// 在dom加载前初始化，onMounted会报错
onBeforeMount(() => {
  // 初始化表单项
  initForm();
});

// onMounted(() => {
//   // 转化表单项 -> checkbox-group
//   transformCheckbox(0);
//
// });
</script>

<style lang="scss" scoped>
:deep(.ant-btn) {
  width: 68px;
  height: 32px;
}

:deep(.ant-select-item-option-selected) {
  background: #eeecfc !important;
  color: #5a47e5 !important;
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

:deep(.ant-slider-handle:hover::after) {
  box-shadow: 0 0 0 4px #5a47e5;
}

:deep(.ant-slider:hover .ant-slider-track) {
  background-color: #5a47e5;
}

.form-item-inline {
  display: flex;
  justify-content: flex-end;

  & .ant-form-item:nth-child(1) {
    flex: 1;

    :deep(.ant-form-item-control-input, .ant-form-item-control) {
      flex: 1;
      margin-left: 25px;
    }
  }

  & .ant-form-item:nth-child(2) {
    display: flex;
    justify-content: flex-end;
  }
}
</style>
