<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-10-30 17:47:34
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-07 20:06:18
 * @FilePath: \ai-demo\src\App.vue
 * @Description: 
-->
<template>
  <a-config-provider :locale="locale" :auto-insert-space-in-button="false">
    <div class="page-app">
      <router-view></router-view>
    </div>
  </a-config-provider>
</template>
<script>
import zhCN from 'ant-design-vue/es/locale/zh_CN';

export default {
  data() {
    return {
      locale: zhCN,
    };
  },
};
</script>

<style lang="scss">
#app {
  margin: 0 auto;
  user-select: text;

  img {
    user-select: none;
  }
}
.page-app {
  background: #fff;
  height: 100vh;
  min-width: 1200px;
  overflow-y: hidden;
  overflow-x: auto;
}
</style>
