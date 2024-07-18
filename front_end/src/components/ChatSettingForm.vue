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
      <a-select
        v-model:value="chatSettingForm.modelType"
        placeholder="请选择你的模型"
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
          {{ item.modelName.length === 0 ? '自定义模型配置' : item.modelName }}
        </a-select-option>
      </a-select>
    </a-form-item>
    <a-form-item
      v-if="chatSettingForm.modelType === '自定义模型配置'"
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
      <a-input-password
        v-model:value="chatSettingForm.apiKey"
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
      />
    </a-form-item>
    <a-form-item ref="apiBase" label="API路径" name="apiBase">
      <a-input v-model:value="chatSettingForm.apiBase" />
    </a-form-item>
    <a-form-item ref="apiModelName" label="模型名称" name="apiModelName">
      <a-input v-model:value="chatSettingForm.apiModelName" />
    </a-form-item>
    <div class="form-item-inline">
      <a-form-item ref="apiContextLength" label="上下文token数" name="apiContextLength">
        <a-slider v-model:value="chatSettingForm.apiContextLength" :min="1" :step="1" />
      </a-form-item>
      <a-form-item name="apiContextLength">
        <a-input-number
          v-model:value="chatSettingForm.apiContextLength"
          :min="1"
          :step="1"
          style="margin-left: 16px"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item ref="maxToken" label="最大token数" name="maxToken">
        <a-slider v-model:value="chatSettingForm.maxToken" :min="1" :step="1" />
      </a-form-item>
      <a-form-item name="maxToken">
        <a-input-number
          v-model:value="chatSettingForm.maxToken"
          :min="1"
          :step="1"
          style="margin-left: 16px"
        />
      </a-form-item>
    </div>
    <div class="form-item-inline">
      <a-form-item ref="context" :label="`上下文数量（${contextLength}）`" name="context">
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
    <a-form-item label="模型能力" name="capabilities">
      <a-checkbox-group v-model:value="capabilitiesOptionsState" style="width: 100%">
        <a-row>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.networkSearch }}</p>
              </template>
              <a-checkbox value="onlineSearch">联网检索</a-checkbox>
            </a-popover>
          </a-col>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.mixedSearch }}</p>
              </template>
              <a-checkbox value="mixedSearch">混合检索</a-checkbox>
            </a-popover>
          </a-col>
          <a-col :span="8">
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.onlySearch }}</p>
              </template>
              <a-checkbox value="onlySearch">仅检索模式</a-checkbox>
            </a-popover>
          </a-col>
        </a-row>
      </a-checkbox-group>
    </a-form-item>
    <a-form-item v-if="chatSettingForm.modelType === '自定义模型配置'" :wrapper-col="{ span: 24 }">
      <a-button type="primary" html-type="submit" style="margin-left: 0 !important; width: auto">
        保存当前自定义模型
      </a-button>
    </a-form-item>
  </a-form>
</template>

<script setup lang="ts">
import { IChatSetting } from '@/utils/types';
import type { Rule } from 'ant-design-vue/es/form';
import { useChat } from '@/store/useChat';
import { useHomeChat } from '@/store/useHomeChat';
import { getLanguage } from '@/language';
import { message } from 'ant-design-vue';

const common = getLanguage().common;

const { useHomeChatSetting } = useChat();
const { contextLength } = storeToRefs(useHomeChat());

const { chatSettingConfigured } = storeToRefs(useHomeChatSetting());
const { setChatSettingConfigured } = useHomeChatSetting();

const formRef = ref(null);
const capabilitiesOptionsState = ref([]);

defineExpose({ onCheck });

const chatSettingForm = ref<IChatSetting>();

const rules: Record<string, Rule[]> = {
  modelType: [{ required: true, message: '请选择模型提供方', trigger: 'change' }],
  modelName: [{ required: true, message: '请输入当前自定义模型名称', trigger: 'change' }],
  apiKey: [{ required: true, message: '请输入密钥', trigger: 'change' }],
  apiBase: [{ required: true, message: '请输入API地址', trigger: 'change' }],
  apiModelName: [{ required: true, message: '请输入模型名称', trigger: 'change' }],
  apiContextLength: [{ required: true, message: '请输入上下文token数', trigger: 'change' }],
  maxToken: [{ required: true, message: '请输入maxToken', trigger: 'change' }],
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
