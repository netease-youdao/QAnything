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
    <div class="header-navs">
      <div
        v-for="item in navList"
        :key="item.name"
        :class="['nav-item', navIndex === item.value ? 'nav-item-active' : '']"
        @click="setNavIdx(item.value)"
      >
        <!--        <div :class="['item-icon', item.value === 0 ? 'knowledge-icon' : 'bot-icon']"></div>-->
        <div :class="['item-icon', getIcon(item.value)]"></div>
        {{ item.name }}
      </div>
    </div>
    <ul>
      <!-- <li @click="goDetail('https://ai.youdao.com/qanything.s')">
        <img src="../assets/home/icon-home.png" alt="首页" /><span>首页</span>
      </li>
      <li><img src="../assets/home/icon-document.png" alt="开发文档" /><span>开发文档</span></li> -->
      <li class="toggle-button">
        <span :class="[language === 'zh' ? 'active' : '']" @click="changLanguage('zh')">中</span>
        <span class="line"></span>
        <span :class="[language === 'en' ? 'active' : '']" @click="changLanguage('en')">En</span>
      </li>
      <li>
        <a-popover placement="bottomRight" overlay-class-name="cooperate">
          <template #content>
            <p>Aldoud_Business@corp.youdao.com</p>
          </template>
          <template #title>
            <span>{{ header.cooperationMore }}</span>
          </template>
          <div class="myspan">
            <img src="../assets/home/icon-email.png" alt="合作咨询" /><span>{{
              header.cooperation
            }}</span>
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
import { useHeader } from '@/store/useHeader';
import { useLanguage } from '@/store/useLanguage';
import { getLanguage } from '@/language/index';
import routeController from '@/controller/router';

const header = getLanguage().header;
const { language } = storeToRefs(useLanguage());
const { navIndex } = storeToRefs(useHeader());
const { setLanguage } = useLanguage();
const { setNavIndex } = useHeader();
const { changePage } = routeController();

const navList = [
  {
    name: getLanguage().header.quickStart,
    value: 0,
  },
  {
    name: getLanguage().header.knowledge,
    value: 1,
  },
  {
    name: 'Bots',
    value: 2,
  },
];

const changLanguage = (lang: string) => {
  setLanguage(lang);
  window.location.reload();
};

const goDetail = (url: string) => {
  console.log(url);
  window.location.href = url;
};

const setNavIdx = value => {
  if (navIndex.value === value) {
    return;
  }
  setNavIndex(value);
  if (value === 1) {
    changePage('/home');
  } else if (value === 2) {
    changePage('/bots');
  }
};

// header的item-icon选择
const iconMap = new Map([
  [0, 'quick-icon'],
  [1, 'knowledge-icon'],
  [2, 'bot-icon'],
]);
const getIcon = itemValue => {
  return iconMap.get(itemValue);
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

  .header-navs {
    //width: 234px;
    height: 50px;
    margin-left: 158px;
    display: flex;
    flex: 1;
    //justify-content: space-between;
    .nav-item {
      //width: 80px;
      height: 50px;
      margin-right: 40px;
      color: #999999;
      font-size: 18px;
      display: flex;
      align-items: center;
      cursor: pointer;

      img {
        width: 20px;
        height: 20px;
        margin-right: 4px;
      }

      .item-icon {
        width: 20px;
        height: 20px;
        margin-right: 4px;
        background-size: cover;
        background-repeat: no-repeat;
      }

      .bot-icon {
        background-image: url('@/assets/header/bots-icon.png');
      }

      .knowledge-icon {
        background-image: url('@/assets/header/knowledge-icon.png');
      }

      .quick-icon {
        background-image: url('@/assets/header/quick-icon.png');
      }
    }

    .nav-item-active {
      color: #fff;

      .bot-icon {
        background-image: url('@/assets/header/bots-active-icon.png');
      }

      .knowledge-icon {
        background-image: url('@/assets/header/knowledge-active-icon.png');
      }

      .quick-icon {
        background-image: url('@/assets/header/quick-active-icon.png');
      }
    }
  }

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

  .toggle-button {
    font-size: 14px;
    font-weight: 300;
    line-height: 22px;
    color: #cccccc;
    cursor: pointer;

    .active {
      color: #ffffff;
      font-weight: 500;
    }

    .line {
      width: 1px;
      height: 14px;
      border-left: 1px solid rgb(216, 216, 216, 0.3);
      margin: 0px 8px;
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
