<template>
  <a-config-provider :theme="{ token: { colorPrimary: '#5a47e5' } }">
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
      <a-form-item ref="apiBase" :label="common.apiBaseLabel" name="apiBase">
        <a-input v-model:value="chatSettingForm.apiBase" aria-autocomplete="none" />
      </a-form-item>
      <a-form-item ref="apiModelName" :label="common.apiModelNameLabel" name="apiModelName">
        <a-input
          v-if="chatSettingForm.modelType !== 'openAI'"
          v-model:value="chatSettingForm.apiModelName"
          aria-autocomplete="none"
        />
        <a-select
          v-else
          v-model:value="chatSettingForm.apiModelName"
          :options="openAIModelDefault.map(item => ({ value: item }))"
          @change="openAIModelSelect"
        >
          <template #dropdownRender="{ menuNode: menu }">
            <v-nodes :vnodes="menu" />
            <a-divider style="margin: 4px 0" />
            <a-space style="padding: 4px 8px">
              <a-input ref="inputRef" v-model:value="name" placeholder="输入模型名称" />
              <a-button type="text" @click="addItem">添加模型</a-button>
            </a-space>
          </template>
        </a-select>
      </a-form-item>
      <div class="form-item-inline">
        <a-form-item ref="apiContextLength" name="apiContextLength">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.apiContextLengthLabelDescription }}</p>
              </template>
              <span>{{ common.apiContextLengthLabel }}</span>
            </a-popover>
          </template>
          <a-slider
            v-model:value="apiContextTokenK"
            :min="4"
            :max="openAIModelMax || 200"
            :step="1"
            :tip-formatter="(value: number) => `${value}K`"
          />
        </a-form-item>
        <a-form-item name="apiContextLength">
          <a-input-number
            v-model:value="apiContextTokenK"
            :min="4"
            :max="openAIModelMax || 200"
            :step="1"
            style="margin-left: 16px"
            :precision="0"
            :controls="false"
            addon-after="K"
          />
        </a-form-item>
      </div>
      <div class="form-item-inline">
        <a-form-item ref="maxToken" name="maxToken">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.maxTokenLabelDescription }}</p>
              </template>
              <span>{{ common.maxTokenLabel }}</span>
            </a-popover>
          </template>
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
            :controls="false"
          />
        </a-form-item>
      </div>
      <div class="form-item-inline">
        <a-form-item ref="chunkSize" name="chunkSize">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.chunkSizeLabelDescription }}</p>
              </template>
              <span>{{ common.chunkSizeLabel }}</span>
            </a-popover>
          </template>
          <a-slider
            v-model:value="chatSettingForm.chunkSize"
            :min="400"
            :max="chatSettingForm.apiContextLength / TOKENRATIO"
            :step="1"
          />
        </a-form-item>
        <a-form-item name="chunkSize">
          <a-input-number
            v-model:value="chatSettingForm.chunkSize"
            :min="400"
            :max="chatSettingForm.apiContextLength / TOKENRATIO"
            :step="1"
            style="margin-left: 16px"
            :precision="0"
            :controls="false"
          />
        </a-form-item>
      </div>
      <div class="form-item-inline">
        <a-form-item ref="temperature" name="temperature">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.temperatureLabelDescription }}</p>
              </template>
              <span>{{ common.temperatureLabel }}</span>
            </a-popover>
          </template>
          <a-slider v-model:value="chatSettingForm.temperature" :min="0" :max="1" :step="0.01" />
        </a-form-item>
        <a-form-item name="temperature">
          <a-input-number
            v-model:value="chatSettingForm.temperature"
            :min="0"
            :max="1"
            :step="0.01"
            style="margin-left: 16px"
            :precision="2"
            :controls="false"
          />
        </a-form-item>
      </div>
      <div class="form-item-inline">
        <a-form-item ref="top_P" name="top_P">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.top_PLabelDescription }}</p>
              </template>
              <span>{{ common.top_PLabel }}</span>
            </a-popover>
          </template>
          <a-slider v-model:value="chatSettingForm.top_P" :min="0" :max="1" :step="0.01" />
        </a-form-item>
        <a-form-item name="top_P">
          <a-input-number
            v-model:value="chatSettingForm.top_P"
            :min="0"
            :max="1"
            :step="0.01"
            style="margin-left: 16px"
            :precision="2"
            :controls="false"
          />
        </a-form-item>
      </div>
      <div class="form-item-inline">
        <a-form-item ref="top_K" name="top_K">
          <template #label>
            <a-popover placement="topLeft">
              <template #content>
                <p>{{ common.top_KLabelDescription }}</p>
              </template>
              <span>{{ common.top_KLabel }}</span>
            </a-popover>
          </template>
          <a-slider v-model:value="chatSettingForm.top_K" :min="1" :max="100" :step="1" />
        </a-form-item>
        <a-form-item name="top_K">
          <a-input-number
            v-model:value="chatSettingForm.top_K"
            :min="1"
            :max="100"
            :step="1"
            style="margin-left: 16px"
            :precision="0"
            :controls="false"
          />
        </a-form-item>
      </div>
      <a-form-item ref="context" name="context">
        <template #label>
          <a-popover placement="topLeft">
            <template #content>
              <p>{{ common.contextLabelDescription }}</p>
            </template>
            <span>{{ common.contextLabel }}</span>
          </a-popover>
        </template>
        <a-slider
          v-model:value="chatSettingForm.context"
          :min="0"
          :max="11"
          :step="1"
          :tip-formatter="sliderFormatter"
        />
      </a-form-item>
      <a-form-item :label="common.capabilitiesLabel" name="capabilities">
        <a-checkbox-group v-model:value="capabilitiesOptionsState" style="width: 100%">
          <a-row>
            <a-col v-for="key in Object.keys(chatSettingForm.capabilities)" :key="key" :span="6">
              <a-popover placement="topLeft">
                <template #content>
                  <p>{{ common[`${key}Description`] }}</p>
                </template>
                <a-checkbox :value="key">{{ common[key] }}</a-checkbox>
              </a-popover>
            </a-col>
          </a-row>
        </a-checkbox-group>
      </a-form-item>
      <a-form-item
        v-if="chatSettingForm.modelType === '自定义模型配置'"
        :wrapper-col="{ span: 24 }"
      >
        <a-button type="primary" html-type="submit" style="margin-left: 0 !important; width: auto">
          {{ common.saveModel }}
        </a-button>
      </a-form-item>
    </a-form>
  </a-config-provider>
</template>

<script setup lang="ts">
import { IChatSetting } from '@/utils/types';
import type { Rule } from 'ant-design-vue/es/form';
import { useChatSetting } from '@/store/useChatSetting';
import { getLanguage } from '@/language';
import { message } from 'ant-design-vue';

defineExpose({ onCheck });

const common = getLanguage().common;

const { chatSettingConfigured } = storeToRefs(useChatSetting());
const { setChatSettingConfigured, openAISettingMap } = useChatSetting();

const formRef = ref(null);
const capabilitiesOptionsState = ref([]);

const chatSettingForm = ref<IChatSetting>();

// 如果是openAI，做选择操作，默认填入model和apiContextLength，但是也需要自定义添加
const VNodes = defineComponent({
  props: {
    vnodes: {
      type: Object,
      required: true,
    },
  },
  render() {
    return this.vnodes;
  },
});
let index = 0;
const openAIModelDefault = ref([]);
const inputRef = ref();
const name = ref();

// openai选中的话需要有最大值为默认值
const openAIModelMax = ref(200);

const addItem = e => {
  e.preventDefault();
  openAIModelDefault.value.push(name.value || `New item ${(index += 1)}`);
  name.value = '';
  setTimeout(() => {
    inputRef.value?.focus();
  }, 0);
};

onMounted(() => {
  openAIModelDefault.value = [...openAISettingMap.keys()];
});

const openAIModelSelect = (value: any) => {
  const curContextLength = openAISettingMap.get(value)?.apiContextLength;
  if (curContextLength) {
    apiContextTokenK.value = curContextLength / 1024;
    openAIModelMax.value = curContextLength / 1024;
  } else {
    apiContextTokenK.value = 4;
  }
};

// maxToken是上下文token的 1/TOKENRATIO 倍
const TOKENRATIO = 4;

// 上下文条数，无限制tooltip
const sliderFormatter = (value: number) => {
  // 11代表无限制，所以最后发送的时候如果是11，需要做无限制处理
  return value >= 11 ? '无限制' : value;
};

// 上下文长度，单位k，绑定到模板上
const apiContextTokenK = ref(4);
onMounted(() => {
  apiContextTokenK.value = chatSettingForm.value.apiContextLength / 1024;
});
// 存储需要字节，* 1024
watch(
  () => apiContextTokenK.value,
  () => {
    chatSettingForm.value.apiContextLength = apiContextTokenK.value * 1024;
  }
);

const rules: Record<string, Rule[]> = {
  modelType: [
    {
      required: true,
      message: `${common.plsInput}${common.modelProviderLabel}`,
      trigger: 'change',
    },
  ],
  modelName: [
    { required: true, message: `${common.plsInput}${common.modelNameLabel}`, trigger: 'change' },
  ],
  apiKey: [
    { required: true, message: `${common.plsInput}${common.apiKeyLabel}`, trigger: 'change' },
  ],
  apiBase: [
    { required: true, message: `${common.plsInput}${common.apiBaseLabel}`, trigger: 'change' },
  ],
  apiModelName: [
    {
      required: true,
      message: `${common.plsInput}${common.apiModelNameLabel}`,
      trigger: 'change',
    },
  ],
  apiContextLength: [
    {
      required: true,
      message: `${common.plsInput}${common.apiContextLengthLabel}`,
      trigger: 'change',
    },
  ],
  maxToken: [
    { required: true, message: `${common.plsInput}${common.maxTokenLabel}`, trigger: 'change' },
  ],
  chunkSize: [
    { required: true, message: `${common.plsInput}${common.chunkSizeLabel}`, trigger: 'change' },
  ],
  temperature: [{ required: true, message: `${common.plsInput}temperature`, trigger: 'change' }],
  top_P: [{ required: true, message: `${common.plsInput}top_P`, trigger: 'change' }],
  top_K: [{ required: true, message: `${common.plsInput}top_K`, trigger: 'change' }],
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
  openAIModelMax.value = 200;
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
  // 如果是openai，就设置最大值
  if (activeForm.modelType === 'openAI' && openAISettingMap.has(activeForm.apiModelName as any)) {
    openAIModelSelect(activeForm.apiModelName);
  }
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
</script>

<style lang="scss" scoped>
//:deep(.ant-btn) {
//  width: 68px;
//  height: 32px;
//}

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

////hover
//:deep(
//    :where(.css-dev-only-do-not-override-19iuou).ant-checkbox-wrapper:not(
//        .ant-checkbox-wrapper-disabled
//      ):hover
//      .ant-checkbox-inner,
//    :where(.css-dev-only-do-not-override-19iuou).ant-checkbox:not(.ant-checkbox-disabled):hover
//      .ant-checkbox-inner
//  ) {
//  border-color: #5a47e5 !important;
//}
//
//// 选中hover
//:deep(
//    :where(.css-dev-only-do-not-override-19iuou).ant-checkbox-wrapper:not(
//        .ant-checkbox-wrapper-disabled
//      ):hover
//      .ant-checkbox-checked:not(.ant-checkbox-disabled)
//      .ant-checkbox-inner
//  ) {
//  background-color: #5a47e5;
//  border-color: #5a47e5 !important;
//}
//
//// 选中外圈hover
//:deep(
//    :where(.css-dev-only-do-not-override-19iuou).ant-checkbox-wrapper:not(
//        .ant-checkbox-wrapper-disabled
//      ):hover
//      .ant-checkbox-checked:not(.ant-checkbox-disabled):after
//  ) {
//  border-color: #5a47e5 !important;
//}
//
//// 选中正常
//:deep(:where(.css-dev-only-do-not-override-19iuou).ant-checkbox-checked .ant-checkbox-inner) {
//  background-color: #5a47e5;
//  border-color: #5a47e5 !important;
//}
//
//:deep(
//    :where(.css-dev-only-do-not-override-19iuou).ant-checkbox-checked:not(
//        .ant-checkbox-disabled
//      ):hover
//      .ant-checkbox-inner
//  ) {
//  background-color: #5a47e5 !important;
//}
//
////background-color: #5a47e5;
////border-color: #5a47e5 !important;
//
:deep(.ant-slider-handle:hover::after) {
  box-shadow: 0 0 0 4px #5a47e5;
}

//:deep(.ant-slider-handle::after) {
//  box-shadow: 0 0 0 4px #5a47e5;
//}

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
    width: 106px;
  }

  :deep(.ant-form-item-explain-error) {
    padding-left: 25px;
  }
}
</style>
