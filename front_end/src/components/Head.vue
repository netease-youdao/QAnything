<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-10 15:02:33
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-24 10:19:43
 * @FilePath: front_end/src/components/Head.vue
 * @Description: 
-->
<template>
  <div class="header">
    <div class="logo">
      <img
        src="../assets/login/logo-small.png"
        :style="{ marginTop: navIndex === -1 ? '' : '14px' }"
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
        <span>{{ item.name }}</span>
      </div>
    </div>
    <ul>
      <!-- <li @click="goDetail('https://ai.youdao.com/qanything.s')">
        <img src="../assets/home/icon-home.png" alt="首页" /><span>首页</span>
      </li>
      <li><img src="../assets/home/icon-document.png" alt="开发文档" /><span>开发文档</span></li> -->
      <li class="phone">
        <a-popover>
          <template #content>
            <p>仅做用户标识作用, 不会对用户信息做任何收集</p>
          </template>
          <a-input
            v-model:value="phone"
            size="small"
            :status="phoneStatus"
            style="width: 100%"
            @blur="confirmPhone"
          >
            <template #prefix>
              <PhoneOutlined v-if="phoneStatus !== 'error'" />
              <CloseCircleOutlined v-else />
            </template>
          </a-input>
        </a-popover>
      </li>
      <li class="toggle-button">
        <span :class="[language === 'zh' ? 'active' : '']" @click="changLanguage('zh')">中</span>
        <span class="line"></span>
        <span :class="[language === 'en' ? 'active' : '']" @click="changLanguage('en')">En</span>
      </li>
      <li>
        <a-popover placement="bottomRight">
          <template #content>
            <p>010-82558901（商务）</p>
            <p>qanything@rd.netease.com（技术）</p>
            <p>Aldoud_Business@corp.youdao.com（商务）</p>
          </template>
          <template #title>
            <span>{{ header.cooperationMore }}</span>
          </template>
          <div class="myspan">
            <img src="../assets/home/icon-email.png" alt="合作咨询" />
            <span>{{ header.cooperation }}</span>
          </div>
        </a-popover>
      </li>
      <li>
        <div class="myspan" @click="goStatistics">
          <LineChartOutlined style="margin-right: 5px" />
          <span>{{ header.statistics }}</span>
        </div>
      </li>
    </ul>
  </div>
</template>
<script lang="ts" setup>
// import { useUser } from '@/store/useUser';
// const { userInfo } = storeToRefs(useUser());
import { LineChartOutlined, CloseCircleOutlined, PhoneOutlined } from '@ant-design/icons-vue';
import { useHeader } from '@/store/useHeader';
import { useLanguage } from '@/store/useLanguage';
import { getLanguage } from '@/language/index';
import routeController from '@/controller/router';
import message from 'ant-design-vue/es/message';
import { useUser } from '@/store/useUser';

const header = getLanguage().header;
const { language } = storeToRefs(useLanguage());
const { navIndex } = storeToRefs(useHeader());
const { setLanguage } = useLanguage();
const { setNavIndex } = useHeader();
const { changePage } = routeController();
const { userInfo, setPhoneNumber } = useUser();

const navList = [
  {
    name: getLanguage().header.quickStart,
    value: 2,
  },
  {
    name: getLanguage().header.knowledge,
    value: 0,
  },
  {
    name: 'Bots',
    value: 1,
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
  if (value === 0) {
    changePage('/home');
  } else if (value === 1) {
    changePage('/bots');
  } else if (value === 2) {
    changePage('/quickstart');
  }
};

// header的item-icon选择
const iconMap = new Map([
  [0, 'knowledge-icon'],
  [1, 'bot-icon'],
  [2, 'quick-icon'],
]);
const getIcon = itemValue => {
  return iconMap.get(itemValue);
};

const goStatistics = () => {
  setNavIndex(-1);
  changePage('/statistics');
};

const checkPhoneRegex = (phone: string): boolean => {
  const regex = /^1[3-9]\d{9}$/;
  return regex.test(phone);
};

// 手机号
const phone = ref('');
const phoneStatus = ref('');
const confirmPhone = () => {
  if (!checkPhoneRegex(phone.value)) {
    phoneStatus.value = 'error';
    message.error('请输入正确的手机号');
  } else {
    setPhoneNumber(phone.value);
    phoneStatus.value = '';
    message.success('绑定成功');
    window.location.reload();
  }
};
onMounted(() => {
  phone.value = userInfo.phoneNumber === '1' ? '' : userInfo.phoneNumber;
});
</script>
<style lang="scss" scoped>
.header {
  width: 100vw;
  //min-width: 1200px;
  height: 64px;
  display: flex;
  align-items: center;
  background: #26293b;

  .header-navs {
    //width: 234px;
    height: 50px;
    margin-left: 58px;
    display: flex;
    flex: 1;
    align-items: center;

    .nav-item {
      //width: 100px;
      height: 20px;
      margin-right: 40px;
      display: flex;
      align-items: center;
      font-size: 1rem;
      cursor: pointer;
      color: #999999;

      span {
        white-space: nowrap;
      }

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
    min-width: 280px;
    height: 64px;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;

    img {
      width: 146px;
      height: 28px;
      //margin-top: 14px;
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

  .phone {
    width: 152px;
  }

  ul {
    display: flex;
    margin-left: auto;
    margin-right: 32px;

    li {
      display: flex;
      align-items: center;
      margin-left: 36px;
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

      span {
        font-size: 14px;
        white-space: nowrap;
      }
    }
  }
}
</style>
