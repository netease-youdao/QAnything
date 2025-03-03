<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-29 10:00:43
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 16:08:44
 * @FilePath: front_end/src/components/ChatInfoPanel.vue
 * @Description: ai对话的消耗token和耗时, 还有当时对话的模型信息
 -->
<template>
  <div class="content">
    <a-typography-text type="secondary" style="font-size: 12px">
      {{ outerInfo }}
      <a-tooltip
        placement="bottom"
        color="#666666"
        overlay-class-name="tooltip-class"
        @click="openInfoModal"
      >
        <template #title>
          <span>{{ common.modelInfoView }}</span>
        </template>
        <SvgIcon name="question" />
      </a-tooltip>
    </a-typography-text>
  </div>
</template>

<script setup lang="ts">
import { IChatItemInfo } from '@/utils/types';
import { formatTimestamp } from '@/utils/utils';
import SvgIcon from '@/components/SvgIcon.vue';
import { Modal, TypographyParagraph } from 'ant-design-vue';
import { getLanguage } from '@/language';

const { common } = getLanguage();

interface IProps {
  chatItemInfo: IChatItemInfo;
}

const props = defineProps<IProps>();

const {
  timeInfo: timeInfoOrigin,
  tokenInfo,
  settingInfo: settingInfoOrigin,
  dateInfo: dateInfoOrigin,
} = toRefs(props.chatItemInfo);

// 需要展示的time信息
const TIMEINFO = new Set([
  'preprocess',
  'condense_q_chain',
  'retriever_search',
  'web_search',
  'rerank',
  'reprocess',
  'llm_first_return',
  'first_return',
  'llm_completed',
  'obtain_images_time',
  'chat_completed',
]);

// 需要展示的模型信息
const MODELINFO = new Set([
  'apiBase',
  'apiContextLength',
  'apiKey',
  'apiModelName',
  'context',
  'maxToken',
  'temperature',
  'top_P',
]);

// 外层展示的所有信息
const OUTERINFO = new Set([
  'first_return',
  'chat_completed',
  'total_tokens',
  'prompt_tokens',
  'completion_tokens',
  'Model name',
  '模型名称',
  'date',
]);

// time会有多余的值传过来，所以需要过滤一下
const timeInfo = computed(() => {
  const obj = {};
  for (let i in timeInfoOrigin.value) {
    if (TIMEINFO.has(i)) {
      obj[i] = `${timeInfoOrigin.value[i].toFixed(2)}s`;
    }
  }
  return obj;
});

// 转换模型配置的格式，变为名称，处理模型能力
const settingInfo = computed(() => {
  // 获取名称（中、英）
  const getLabel = (key: string) => {
    return common[key + 'Label'];
  };
  const obj = {};
  for (let i in settingInfoOrigin.value) {
    if (MODELINFO.has(i)) {
      obj[getLabel(i)] = settingInfoOrigin.value[i];
    }
  }
  return obj;
});

// 格式化时间戳，变为2024/8/1 12:30:12 的格式
const dateInfo = computed(() => {
  return formatTimestamp(dateInfoOrigin.value);
});

// 将所有的信息整理到一个对象中
const infoObj = computed(() => {
  return {
    ...timeInfo.value,
    ...tokenInfo.value,
    ...settingInfo.value,
    date: dateInfo.value,
  };
});

// 过滤出外层展示的信息
const outerInfo = computed(() => {
  const obj = {};
  for (let i in infoObj.value) {
    if (OUTERINFO.has(i)) {
      obj[i] = infoObj.value[i];
    }
  }
  return formatInfo(obj);
});

const formatInfo = <T>(obj: T) => {
  return Object.entries(obj)
    .map(([key, value]) => `${key}: ${value}`)
    .join(', ');
};

// 打开详细信息
const openInfoModal = () => {
  const formatTimeInfo = (timeData, keys) => {
    let formattedString = '';
    keys.forEach((key, index) => {
      formattedString += `${key}: ${timeData[key]}`;
      if (index < keys.length - 1) {
        formattedString += ' + ';
      }
    });
    return formattedString;
  };
  Modal.info({
    title: `${common.modelInfoTitle}`,
    content: h('div', { style: { 'user-select': 'text' } }, [
      h(TypographyParagraph, {}, () => `${common.modelInfoTime}: ${formatInfo(timeInfo.value)}`),
      h(
        TypographyParagraph,
        { mark: true },
        () => `${common.note}：${formatTimeInfo(
          timeInfo.value,
          [...TIMEINFO.values()].slice(0, 7)
        )} = first_return: ${timeInfo.value['first_return']}
        + llm_completed：${timeInfo.value['llm_completed']}
        + obtain_images_time: ${timeInfo.value['obtain_images_time'] || '0.00s'}
        = chat_completed：${timeInfo.value['chat_completed']}`
      ),
      h(TypographyParagraph, {}, () => `${common.modelInfoToken}: ${formatInfo(tokenInfo.value)}`),
      h(TypographyParagraph, {}, () => [
        `${common.modelInfoSetting}：`,
        ...Object.entries(settingInfo.value).map(([key, value]) =>
          h(TypographyParagraph, {}, () => `${key}: ${value}`)
        ),
      ]),
    ]),
    width: '30%',
    maskClosable: true,
    centered: true,
  });
};
</script>

<style lang="scss" scoped>
.content {
  svg {
    display: inline-block;
    width: 15px;
    height: 15px;
    padding-top: 2px;
    color: #c1c1c1;

    &:focus {
      outline: none;
    }
  }

  .tooltip-class {
    width: 500px !important;
  }
}
</style>
