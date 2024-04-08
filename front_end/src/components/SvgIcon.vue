<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-06-21 14:45:34
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-01 11:18:03
 * @FilePath: \ai-demo\src\components\SvgIcon.vue
 * @Description: 
-->
<template>
  <svg aria-hidden="true">
    <use :href="symbolId" fill="currentColor" />
  </svg>
</template>

<script>
import { defineComponent, computed } from 'vue';

export default defineComponent({
  name: 'SvgIcon',
  props: {
    prefix: {
      type: String,
      default: 'icon',
    },
    name: {
      type: String,
      required: true,
    },
  },
  setup(props) {
    const symbolId = computed(() => `#${props.prefix}-${props.name}`);
    return { symbolId };
  },
});
</script>
<style>
/* svg {
  width: 28px;
  height: 28px;
} */
</style>
