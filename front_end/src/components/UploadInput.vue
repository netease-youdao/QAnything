<template>
  <div class="mt-50px">
    <div class="basic-box">
      <a-input
        v-model:value="inputVal"
        :status="status"
        :placeholder="common.urlPlaceholder"
        @change="setStatus()"
      >
        <template #suffix>
          <SvgIcon name="add" class="mt3" @click="add"></SvgIcon>
        </template>
      </a-input>
      <span v-show="status === 'error'" class="red-text">{{ common.errTip }}</span>
    </div>
    <a-form :label-col="labelCol" :wrapper-col="wrapperCol">
      <a-form-item
        v-for="(item, index) in urlList"
        :key="index"
        class="relative"
        @mouseenter="changeStatus(index, 'hover')"
        @mouseleave="changeStatus(index, 'default')"
      >
        <a-input v-model:value="item.text" :placeholder="common.urlPlaceholder">
          <template #suffix>
            <span v-if="item.status === 'hover'" class="mt3">
              <!-- <SvgIcon name="card-confirm" class="mr20" @click="parsing(index)"></SvgIcon>
              <SvgIcon name="card-cancel" @click="clear(index)"></SvgIcon> -->
              <SvgIcon name="card-delete" @click="deleteUrl(index)"></SvgIcon>
            </span>
            <span v-else class="mt3"></span>
          </template>
        </a-input>
        <span v-if="item.status === 'parsing'" class="loading-line-box">
          <span
            ref="percentRef"
            class="loading-line"
            :style="{
              width: item.percent + '%',
            }"
          >
          </span>
        </span>
      </a-form-item>
    </a-form>
  </div>
</template>
<script lang="ts" setup>
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
const { urlList } = storeToRefs(useKnowledgeModal());
import { IUrlListItem } from '@/utils/types';
// import urlResquest from '@/services/urlConfig';
// import { message } from 'ant-design-vue';
// import { resultControl } from '@/utils/utils';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;
const labelCol = { span: 10 };
const wrapperCol = { span: 10 };
const percentRef = ref(null);
const timer = ref([]);
const inputVal = ref('');
// const uploadUrlList = ref([]); // 本次上传文件列表
const status = ref('normal');

type inputStatus = 'default' | 'inputing' | 'parsing' | 'success' | 'defeat' | 'hover';

const props = defineProps({
  kbId: {
    type: String,
    required: true,
  },
});
console.log(props.kbId);
// const parseObj = {
//   default: '默认',
//   inputing: '输入状态',
//   parsing: '解析中',
//   success: '解析完成',
//   defeat: '解析完成',
// };

const add = () => {
  if (!inputVal.value) {
    status.value = 'error';
    return;
  }
  const item: IUrlListItem = {
    status: 'default',
    text: inputVal.value,
    percent: 0,
  };
  // urlList.value.push(item);
  urlList.value.push(item);
  inputVal.value = '';
};

const changeStatus = (index: number, status: inputStatus) => {
  console.log(urlList.value);
  urlList.value[index].status = status;
};

const setStatus = () => {
  if (inputVal.value) {
    status.value = 'normal';
  }
};

// const parsing = async (index: number) => {
//   changeStatus(index, 'parsing');
//   timer[index] = setInterval(() => {
//     if (urlList.value[index].percent < 96) {
//       urlList.value[index].percent += 3;
//     }
//   }, 30);
//   try {
//     const res = await urlResquest.uploadUrl({
//       kb_id: props.kbId,
//       url: urlList.value[index].text,
//       mode: 'soft',
//     });
//     if (+res.code === 200) {
//       changeStatus(index, 'success');
//       urlList.value[index].percent = 0;
//       clearInterval(timer[index]);
//     }
//   } catch (e) {
//     message.error(e.msg || '上传失败');
//     changeStatus(index, 'defeat');
//     clearInterval(timer[index]);
//   }

//   console.log(percentRef.value);
// };

// const clear = (index: number) => {
//   changeStatus(index, 'default');
//   urlList.value[index].text = '';
// };

const deleteUrl = (index: number) => {
  urlList.value.splice(index, 1);
};

onBeforeUnmount(() => {
  timer.value.forEach(item => {
    clearInterval(item);
  });
});
</script>
<style lang="scss" scoped>
:deep(.ant-form-item .ant-form-item-control-input) {
  width: 414px;
}

:deep(.ant-input-affix-wrapper) {
  width: 414px;
}

.basic-box {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
}

.red-text {
  color: #ff4d4f;
  font-size: 14px;
}

svg {
  width: 16px;
  height: 16px;
  cursor: pointer;
}
.loading-line-box {
  position: absolute;
  left: 0;
  bottom: 0;
  width: 582px;
  height: 40px;
}
.loading-line {
  position: absolute;
  left: 0;
  bottom: 0;
  height: 40px;
  border-bottom: 2px solid $baseColor;
  border-radius: 0 0 0 6px;
  transition: width 0.5s easy;
}

.mt3 {
  display: inline-block;
  // margin-top: 3px;
  width: 16px;
  height: 16px;
}
.mr20 {
  margin-right: 20px;
}
// .input-add {
// }
</style>
