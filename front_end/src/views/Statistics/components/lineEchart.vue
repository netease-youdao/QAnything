<!--
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-08-23 10:49:35
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-27 14:23:53
 * @FilePath: front_end/src/views/Statistics/components/lineEchart.vue
 * @Description: 支持多条线, xAxis的横坐标的值按照props.list.data[0]的name来读
 -->

<template>
  <div class="line-echart-box">
    <div class="title">
      {{ title }}
      <!--      {{ title }}: {{ totalDate }}-->
      <span class="desc">{{ desc }}</span>
    </div>
    <div v-if="!list.length" class="no-data">暂无数据</div>
    <div v-else ref="chartRef" class="line-echart"></div>
  </div>
</template>
<script lang="ts" setup>
import * as echarts from 'echarts';
import { EChartsOption, SeriesOption } from 'echarts';
import { PropType } from 'vue';

export interface IChartList {
  options: ChartOptionsType;
  data: ChartDataType[];
}

type ChartDataType = {
  name: string;
  value: number | string;
};

type ChartOptionsType = {
  lineColor: string; // 十六进制
  name?: string; // tooltip显示的名称
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
    type: Array as PropType<IChartList[]>,
    default: () => [],
  },
  formatDesc: {
    type: String,
    default: '总量',
  },
});

const chartRef = ref(null);
let chartInstance;

const initChart = () => {
  chartInstance = echarts.init(chartRef.value);
  const seriesHandle = (list: IChartList) => {
    return {
      data: list.data.map(item => item.value),
      type: 'line',
      // symbol: 'circle', //将小圆点改成实心 不写symbol默认空心
      // symbolSize: 8, //小圆点的大小
      name: list.options.name || '',
      itemStyle: {
        color: list.options.lineColor, //小圆点和线的颜色
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
    };
  };
  // 配置项
  const options: EChartsOption = {
    grid: {},
    xAxis: {
      type: 'category',
      data: props.list[0].data.map(item => item.name),
      boundaryGap: false, // 不留白，从原点开始
      axisLabel: {
        show: true, // 确保标签显示
        interval: 0, // 显示所有标签，如果没有特别的需求
        // formatter: function (value) {
        //   // 检查文本长度是否超过10px，这里的10假设是你的字体大小
        //   // 如果不是，请根据你的字体大小调整比例
        //   if (value.toString().length * 10 > 50) {
        //     // 假设字体大小为12px
        //     return value.toString().substring(0, 10) + '...'; // 显示前4个字符加省略号
        //   }
        //   return value;
        // },
      },
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
      // formatter: function (params) {
      //   let str = params[0].axisValueLabel + '<br />';
      //   params.forEach(item => {
      //     str +=
      //       '<span style="display:inline-block; margin-right:5px; width:8px; height:8px; left:8px; background-color:' +
      //       item.color +
      //       '"></span>' +
      //       props.formatDesc +
      //       ' : ' +
      //       params[0].value +
      //       '</span><br/>';
      //   });
      //   return str;
      // },
      axisPointer: {
        type: 'line', // axisPointer 类型，可选为：'line' | 'shadow' | 'cross'
      },
    },
    lineStyle: {
      // 设置线条的style等
      normal: {
        color: '#7261E9', // 折线线条颜色
      },
    },
    series: props.list.map(item => seriesHandle(item)) as SeriesOption[],
  };
  // 设置图表配置项
  chartInstance.setOption(options);
};

onMounted(() => {
  if (props.list.length) {
    initChart();
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
    overflow: auto;

    &::-webkit-scrollbar {
      height: 12px;
    }
  }
}
</style>
