<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:47:35
 * @FilePath: /QAnything/front_end/src/App.vue
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
