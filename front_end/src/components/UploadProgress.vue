<template>
  <div v-if="dataSource.length !== 0" class="container">
    <span class="title">{{ progress.uploadTotalProgress }}：</span>
    <a-tooltip
      :title="`${progress.putStorage}: ${progressPercentCount.green},
      ${progress.inLine}: ${progressPercentCount.gray},
      ${progress.parsing}: ${progressPercentCount.yellow},
      ${progress.failure}: ${progressPercentCount.red}`"
    >
      <div class="multi-color-progress">
        <div class="progress-outer">
          <div class="progress">
            <a-progress
              :percent="progressLength.green"
              class="progress-segment green"
              status="active"
              :stroke-color="greenColor"
            />
            <a-progress
              :percent="progressLength.red"
              class="progress-segment red"
              status="active"
              :stroke-color="redColor"
            />
            <a-progress
              :percent="progressLength.yellow"
              class="progress-segment yellow"
              status="active"
              :stroke-color="yellowColor"
            />
            <a-progress
              :percent="progressLength.gray"
              class="progress-segment gray"
              status="active"
              :stroke-color="grayColor"
            />
          </div>
        </div>
        <div class="percent">
          <div class="percent-text">{{ progressPercent.green + progressPercent.red }}%</div>
        </div>
      </div>
    </a-tooltip>
  </div>
</template>

<script setup lang="ts">
/*
 * 绿 z-index 101 base1
 * 红 z-index 102 base2 = 红自己 + base1
 * 黄 z-index 103 base3 = 黄自己 + base2
 * 灰 z-index 104 base4 = 灰自己 + base3
 */
import { useOptiionList } from '@/store/useOptiionList';
import { getLanguage } from '@/language';

type Status = 'green' | 'yellow' | 'red' | 'gray';
type Percent = {
  [K in Status]: number;
};

const greenColor = '#52c41a'; // 成功入库
const yellowColor = '#FFEB3B'; // 解析中
const redColor = '#f5222d'; // 入库失败、解析失败
const grayColor = '#bfbfbf'; // 已上传，正在入库（排队中）

const { dataSource, totalStatus, kbTotal } = storeToRefs(useOptiionList());

const progress = getLanguage().progress;

// 每个颜色的数量
const progressPercentCount = ref<Percent>({
  green: 0,
  yellow: 0,
  red: 0,
  gray: 0,
});

// 每个颜色的百分比
const progressPercent = ref<Percent>({
  green: 0,
  yellow: 0,
  red: 0,
  gray: 0,
});

// 每个颜色的长度
const progressLength = ref<Percent>({
  green: 0,
  yellow: 0,
  red: 0,
  gray: 0,
});

// 计算百分比函数
const computedPercent = () => {
  // 先计算每个状态的总数
  progressPercentCount.value = totalStatus.value;
  // 计算每个状态的百分比，并保留两位小数
  for (const status in progressPercentCount.value) {
    const percent = (progressPercentCount.value[status] / kbTotal.value) * 100;
    progressPercent.value[status as Status] = parseFloat(percent.toFixed(2));
  }

  // 调整百分比，确保总和为 100%
  let totalPercent = Object.values(progressPercent.value).reduce((acc, val) => acc + val, 0);
  const adjustment = 100 - totalPercent;
  if (adjustment > 0) {
    // 百分比加起来不到100，随便调整一个颜色，这里是灰色
    progressPercent.value.gray += adjustment;
  }

  // 计算每个的长度
  /*
   * 绿 z-index 101 base1
   * 红 z-index 102 base2 = 红自己 + base1
   * 黄 z-index 103 base3 = 黄自己 + base2
   * 灰 z-index 104 base4 = 灰自己 + base3
   */
  progressLength.value.green = progressPercent.value.green;
  progressLength.value.red = progressPercent.value.red + progressLength.value.green;
  progressLength.value.yellow = progressPercent.value.yellow + progressLength.value.red;
  progressLength.value.gray = progressPercent.value.gray + progressLength.value.yellow;
};

watch(
  () => totalStatus,
  () => {
    computedPercent();
  },
  {
    immediate: true,
    deep: true,
  }
);
</script>

<style lang="scss" scoped>
.container {
  width: 100%;
  height: 100%;
  position: relative;
}

.title {
  width: 130px;
}

:deep(.ant-tooltip-inner) {
  max-width: 200px;
}

:where(.css-dev-only-do-not-override-19iuou).ant-progress-line {
  margin-bottom: 0;
}

.multi-color-progress {
  width: 100%;
  height: 8px;
  display: flex;
  justify-content: space-between;
}

.progress-outer {
  width: 90%;
  height: 8px;
}

.progress {
  position: relative;
  width: 100%;
  height: 100%;

  :deep(.ant-progress-text) {
    display: none;
  }

  :deep(.ant-progress .ant-progress-inner) {
    background-color: transparent;
  }
}

.percent {
  width: 10%;
  position: relative;
}

.percent-text {
  position: absolute;
  top: 5px;
  width: 2em;
  color: rgba(0, 0, 0, 0.88);
  line-height: 1;
  white-space: nowrap;
  text-align: start;
  vertical-align: middle;
  word-break: normal;
}

.progress-segment {
  position: absolute;
  top: 0;
  left: 0;
  height: 8px;
  border-radius: 4px;
}

/* 为每个颜色段特定的样式 */
.progress-segment.green {
  //width: 45%;
  //left: 35%;
  z-index: 104;
}

.progress-segment.red {
  //width: 10%;
  z-index: 103;
}

.progress-segment.yellow {
  //width: 45%;
  z-index: 102;
}

.progress-segment.gray {
  //width: 35%;
  //left: 10%;
  z-index: 101;
}
</style>
