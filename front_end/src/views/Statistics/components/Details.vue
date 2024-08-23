<template>
  <div>
    <div class="choose-area">
      <!--      <div class="left item-flex">-->
      <!--        <a-range-picker-->
      <!--          v-model:value="dateValue"-->
      <!--          :open="flag"-->
      <!--          @focus="pickerFocus"-->
      <!--          @blur="pickerBlur"-->
      <!--          @change="dateChange"-->
      <!--        />-->

      <!--        <div class="icon" @click="getDeatil">搜索</div>-->
      <!--      </div>-->

      <!--      <div class="right item-flex">-->
      <!--        <div class="export-part butotn" @click="exportPart">导出选中</div>-->
      <!--        <div class="export-all butotn" @click="exportAll">导出全部</div>-->
      <!--      </div>-->
    </div>
  </div>
</template>

<script setup lang="ts">
// 获取全部数据
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';

async function getQADetail() {
  const res = await resultControl(
    await urlResquest.getQAInfo({
      page_id: 10,
    })
  );
  console.log(res);
}

onMounted(() => {
  getQADetail();
});
</script>

<style scoped lang="scss">
.choose-area {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
}

.item-flex {
  display: flex;

  .icon {
    width: 68px;
    height: 32px;
    @include flex-center;
    background: #ffffff;
    border-radius: 6px;
    border: 1px solid #e5e5e5;
    margin-left: 16px;
    font-size: 14px;
    color: #222222;
    cursor: default;

    img {
      width: 20px;
      height: 20px;
    }
  }

  .butotn {
    width: 88px;
    height: 32px;
    line-height: 22px;
    border-radius: 6px;
    opacity: 1;
    background: #ffffff;
    @include flex-center;
    font-size: 14px;
    color: #666666;
    border: 1px solid #e5e5e5;
    cursor: pointer;

    &.export-all {
      color: #ffffff;
      background: #5a47e5;
      margin-left: 16px;
    }
  }
}
</style>
