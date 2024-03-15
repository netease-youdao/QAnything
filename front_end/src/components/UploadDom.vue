<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-09 17:37:50
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-11-29 15:00:09
 * @FilePath: /ai-demo/src/components/UploadDom.vue
 * @Description: 
-->
<template>
  <div class="upload-box">
    <div class="tips" @click="upload">
      <img class="icon-file" src="../assets/home/icon-file.png" alt="图标" />
      <p class="title">{{ home.startDec }}</p>
      <div class="desc-content">
        <p class="desc">{{ home.updesc2 }}</p>
        <p class="desc">{{ home.require1 }}</p>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { getLanguage } from '@/language/index';

const home = getLanguage().home;
// const props = defineProps({
//   show: {
//     require: true,
//     type: Boolean,
//   },
// });

const emits = defineEmits(['update']);
defineProps({
  acceptList: {
    require: true,
    type: Array,
    default: () => {
      return [];
    },
  },
});

// const input = ref(null);

// const change = e => {
//   console.log(e);
//   const files = e.target.files;
//   console.log(files);

//   emits('update', false, files);
// };

const upload = () => {
  emits('update');
};
</script>
<style lang="scss" scoped>
.upload-box {
  position: relative;
  width: 800px;
  height: 340px;
  background: #fff;
  box-sizing: border-box;
  border: 1px dashed #cecece;
  border-radius: 12px;
}

.tips {
  width: 100%;
  height: 100%;
  position: absolute;
  z-index: 10;
  top: 0;
  left: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  .icon-file {
    width: 110px;
    height: 110px;
  }

  .titie {
    margin-top: 12px;
    font-size: 20px;
    font-weight: 600;
    color: $title1;
  }

  .desc-content {
    width: 599px;
    margin: 12px 0 20px 0;
    .desc {
      width: 100%;
      text-align: center;
      font-size: 14px;
      font-weight: normal;
      color: $title2;
    }
  }
}
</style>
