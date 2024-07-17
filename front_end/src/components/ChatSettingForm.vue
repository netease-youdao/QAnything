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
    <a-form-item label="模型提供方" name="modelType">
      <a-select v-model:value="chatSettingForm.modelType" placeholder="please select your zone">
        <a-select-option value="openAi">openAi</a-select-option>
        <a-select-option value="ollama">ollama</a-select-option>
        <a-select-option value="custom">自定义模型配置</a-select-option>
      </a-select>
    </a-form-item>
    <a-form-item
      v-if="chatSettingForm.modelType === 'custom'"
      ref="modelName"
      label="自定义模型名称"
      name="modelName"
    >
      <a-input v-model:value="chatSettingForm.modelName" />
    </a-form-item>
    <a-form-item
      v-if="chatSettingForm.modelType !== 'ollama'"
      ref="apiKey"
      label="API密钥"
      name="apiKey"
    >
      <a-input-password v-model:value="chatSettingForm.apiKey" />
    </a-form-item>
    <a-form-item ref="apiBase" label="API地址" name="apiBase">
      <a-input v-model:value="chatSettingForm.apiBase" />
    </a-form-item>
    <a-form-item ref="apiModelName" label="模型名称" name="apiModelName">
      <a-input v-model:value="chatSettingForm.apiModelName" />
    </a-form-item>
    <a-form-item ref="apiContextLength" label="上下文token数" name="apiContextLength">
      <a-input v-model:value="chatSettingForm.apiContextLength" />
    </a-form-item>
    <a-form-item ref="maxToken" label="最大token数" name="maxToken">
      <a-input v-model:value="chatSettingForm.maxToken" />
    </a-form-item>
    <a-form-item ref="context" label="上下文数量" name="context">
      <a-row>
        <a-col :span="12">
          <a-slider
            v-model:value="chatSettingForm.context"
            :min="0"
            :max="contextLength"
            :step="1"
          />
        </a-col>
        <a-col :span="4">
          <a-input-number
            v-model:value="chatSettingForm.context"
            :min="0"
            :max="contextLength"
            :step="1"
            style="margin-left: 16px"
          />
        </a-col>
      </a-row>
    </a-form-item>
    <a-form-item ref="temperature" label="Temperature" name="temperature">
      <a-row>
        <a-col :span="12">
          <a-slider v-model:value="chatSettingForm.temperature" :min="0" :max="1" :step="0.01" />
        </a-col>
        <a-col :span="4">
          <a-input-number
            v-model:value="chatSettingForm.temperature"
            :min="0"
            :max="1"
            :step="0.01"
            style="margin-left: 16px"
          />
        </a-col>
      </a-row>
    </a-form-item>
    <a-form-item ref="top_P" label="top_P" name="top_P">
      <a-row>
        <a-col :span="12">
          <a-slider v-model:value="chatSettingForm.top_P" :min="0" :max="1" :step="0.01" />
        </a-col>
        <a-col :span="4">
          <a-input-number
            v-model:value="chatSettingForm.top_P"
            :min="0"
            :max="1"
            :step="0.01"
            style="margin-left: 16px"
          />
        </a-col>
      </a-row>
    </a-form-item>
    <a-form-item label="模型能力" name="capabilities">
      <a-checkbox-group v-model:value="capabilitiesOptionsState" :options="capabilitiesOptions" />
    </a-form-item>
    <a-form-item :wrapper-col="{ span: 14, offset: 4 }">
      <a-button type="primary" html-type="submit">保存</a-button>
    </a-form-item>
  </a-form>
</template>

<script setup lang="ts">
import { IChatSetting } from '@/utils/types';
import type { Rule } from 'ant-design-vue/es/form';

const props = defineProps({
  contextLength: {
    type: Number,
    require: true,
    default: 0,
  },
});
const contextLength = ref(props.contextLength);

const capabilitiesOptions = [
  { label: '联网检索', value: 'onlineSearch' },
  { label: '混合检索', value: 'mixedSearch' },
  { label: '仅检索模式', value: 'onlySearch' },
];

const capabilitiesOptionsState = reactive([]);

const chatSettingForm = ref<IChatSetting>({
  modelType: 'openAI', // 默认openAi
  customId: 0,
  modelName: '',
  apiKey: '',
  apiBase: '',
  apiModelName: '',
  apiContextLength: 0,
  maxToken: 0,
  context: 0,
  temperature: 0.5,
  top_P: 1,
  capabilities: {
    onlineSearch: false,
    mixedSearch: false,
    onlySearch: false,
  },
});

// 校验上下文，不得超过最小条数
const validatePassContext = async (_rule: Rule, value: string) => {
  if (value === '') {
    return Promise.reject('请输入上下文条数');
  } else if (Number(value) > contextLength.value) {
    return Promise.reject(`不得超过${contextLength.value}条`);
  } else {
    return Promise.resolve();
  }
};

const onSubmit = (...arg) => {
  console.log('submit');
  console.log(arg);
};

const rules: Record<string, Rule[]> = {
  modelType: [{ required: true, message: '请选择模型提供方', trigger: 'change' }],
  modelName: [{ required: true, message: '请选择模型提供方', trigger: 'change' }],
  apiKey: [{ required: true, message: '请输入密钥', trigger: 'change' }],
  apiBase: [{ required: true, message: '请输入API地址', trigger: 'change' }],
  apiModelName: [{ required: true, message: '请输入模型名称', trigger: 'change' }],
  apiContextLength: [{ required: true, message: '请输入上下文token数', trigger: 'change' }],
  maxToken: [{ required: true, message: '请输入maxToken', trigger: 'change' }],
  context: [{ validator: validatePassContext, trigger: 'change' }],
};

// 转化checkbox-group和自己定义的表单项，0为表单项 -> checkbox-group（初始化用）
function transformCheckbox(type: 0 | 1) {
  if (type === 0) {
    const capabilities = chatSettingForm.value.capabilities;
    for (let item in capabilities) {
      if (capabilities[item]) {
        capabilitiesOptionsState.push(item);
      }
    }
  } else {
    for (let item in chatSettingForm.value.capabilities) {
      chatSettingForm.value.capabilities[item] = false;
    }
    capabilitiesOptionsState.forEach(item => {
      chatSettingForm.value.capabilities[item] = true;
    });
  }
}

// 监听多选框，转化到form表单项中
watch(
  () => capabilitiesOptionsState,
  () => {
    transformCheckbox(1);
  }
);
onMounted(() => {
  // 转化表单项 -> checkbox-group
  transformCheckbox(0);
});
</script>

<style lang="scss" scoped></style>
