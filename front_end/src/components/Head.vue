<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-10 15:02:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-26 11:36:00
 * @FilePath: /qanything-open-source/src/components/Head.vue
 * @Description: 
-->
<template>
  <div class="header">
    <div class="logo">
      <img
        src="../assets/login/logo-small.png"
        alt="logo"
        @click="goDetail('https://ai.youdao.com/')"
      />
    </div>
    <ul>
      <!-- <li @click="goDetail('https://ai.youdao.com/qanything.s')">
        <img src="../assets/home/icon-home.png" alt="首页" /><span>首页</span>
      </li>
      <li><img src="../assets/home/icon-document.png" alt="开发文档" /><span>开发文档</span></li> -->
      <li>
        <a-popover placement="bottomRight" overlay-class-name="cooperate">
          <template #content>
            <p>Aldoud_Business@corp.youdao.com</p>
          </template>
          <template #title>
            <span>更多合作请咨询</span>
          </template>
          <div class="myspan">
            <img src="../assets/home/icon-email.png" alt="合作咨询" /><span>合作咨询</span>
          </div>
        </a-popover>
      </li>
    </ul>
    <div class="user">
      <img src="../assets/home/avatar.png" alt="头像" />
    </div>
  </div>
</template>
<script lang="ts" setup>
// import { useUser } from '@/store/useUser';
// const { userInfo } = storeToRefs(useUser());

const goDetail = (url: string) => {
  console.log(url);
  window.location.href = url;
};
</script>
<style lang="scss" scoped>
.header {
  width: 100vw;
  min-width: 1200px;
  height: 64px;
  display: flex;
  align-items: center;
  background: #26293b;

  .logo {
    width: 146px;
    height: 28px;
    margin-left: 32px;
    cursor: pointer;

    img {
      width: 100%;
      height: 100%;
    }
  }

  ul {
    display: flex;
    margin-left: auto;
    margin-right: 32px;

    li {
      display: flex;
      align-items: center;
      margin-left: 56px;
      color: #ffffff;
      cursor: pointer;

      img {
        width: 16px;
        height: 16px;
        margin-right: 4px;
      }
    }

    .myspan {
      display: flex !important;
      align-items: center;
    }
  }

  .user {
    margin-right: 20px;
    width: 32px;
    height: 32px;
  }
}
</style>
