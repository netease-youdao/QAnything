<!--
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:43:58
 * @FilePath: /QAnything/front_end/src/layout/index.vue
 * @Description: 
-->

<template>
  <a-layout>
    <Head />
    <a-layout>
      <a-layout-sider width="280" style="background: #fff">
        <Sider />
      </a-layout-sider>
      <a-layout>
        <a-layout-content class="main">
          <router-view></router-view>
        </a-layout-content>
      </a-layout>
    </a-layout>
  </a-layout>
</template>
<script lang="ts" setup>
import Head from '@/components/Head.vue';
import Sider from '@/components/Sider.vue';
</script>
<style lang="scss" scoped></style>
