<template>
  <div class="line-echart-box">
    <div class="title">
      {{ title }}: {{ totalDate }}
      <span class="desc">{{ desc }}</span>
    </div>
    <div v-if="!list.length" class="no-data">暂无数据</div>
    <div v-else ref="chartRef" class="line-echart"></div>
  </div>
</template>
<script lang="ts" setup>
import * as echarts from 'echarts';
import { PropType } from 'vue';

export type chartListType = {
  name: string;
  value: number | string;
};

const props = defineProps({
  title: {
    type: String,
    default: '用户',
  },
  desc: {
    type: String,
    default: '',
  },
  list: {
    type: Array as PropType<chartListType[]>,
    default: () => [],
  },
  formatDesc: {
    type: String,
    default: '总量',
  },
});

const chartRef = ref(null);
const totalDate = ref(0);
let chartInstance;

onMounted(() => {
  console.log('lisctlisct', props.list);
  if (props.list.length) {
    chartInstance = echarts.init(chartRef.value);
    // 配置项
    const options = {
      xAxis: {
        type: 'category',
        data: props.list.map(item => item.name),
        // boundaryGap: false, // 不留白，从原点开始
      },
      yAxis: {
        type: 'value',
        splitLine: {
          //网格线
          lineStyle: {
            type: 'dashed', //设置网格线类型 dotted：虚线   solid:实线
          },
        },
      },
      tooltip: {
        trigger: 'axis', // 触发类型，可选为：'item' | 'axis'
        formatter: function (params) {
          totalDate.value = params[0].value;
          let str = params[0].axisValueLabel.slice(0, 10) + '<br />';
          params.forEach(item => {
            str +=
              '<span style="display:inline-block; margin-right:5px; width:8px; height:8px; left:8px; background-color:' +
              item.color +
              '"></span>' +
              props.formatDesc +
              ' : ' +
              params[0].value +
              '<br />';
          });
          return str;
        },
        axisPointer: {
          type: 'line', // axisPointer 类型，可选为：'line' | 'shadow' | 'cross'
        },
      },
      series: [
        {
          data: props.list.map(item => item.value),
          type: 'line',
          // symbol: 'circle', //将小圆点改成实心 不写symbol默认空心
          // symbolSize: 8, //小圆点的大小
          itemStyle: {
            color: '#7261E9', //小圆点和线的颜色
          },
          lineStyle: {
            // 设置线条的style等
            normal: {
              color: '#7261E9', // 折线线条颜色:红色
            },
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {
                offset: 0,
                color: 'rgba(216, 216, 216, 0.3)',
              },
              {
                offset: 1,
                color: 'rgba(114, 97, 233, 0)',
              },
            ]),
          },
        },
      ],
    };
    // 设置图表配置项
    chartInstance.setOption(options);
  }
});

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
});
</script>
<style lang="scss" scoped>
.line-echart-box {
  position: relative;
  width: 100%;
  height: 322px;
  border-radius: 12px;
  opacity: 1;
  background: #ffffff;
  margin-top: 20px;
  padding: 24px 48px 24px 24px;
  box-sizing: border-box;

  .title {
    font-family: PingFang SC;
    font-size: 16px;
    font-weight: 600;
    line-height: 24px;
    letter-spacing: 0;
    color: #222222;

    .desc {
      font-size: 14px;
      margin-left: 12px;
      font-weight: normal;
      line-height: 24px;
      color: #666666;
    }
  }

  .no-data {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: #bfbfbf;
  }

  .line-echart {
    width: 100%;
    height: 100%;
  }
}
</style>
