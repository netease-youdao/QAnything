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
<script lang="ts">
import zhCN from 'ant-design-vue/es/locale/zh_CN';

export default {
  data() {
    return {
      locale: zhCN,
    };
  },
};
</script>
<script setup lang="ts">
import { useLogin } from '@/store/useLogin';
import { useRouter } from 'vue-router';
import localStorage from '@/utils/localStorage';
import urlResquest from '@/services/loginUrlConfig';

const route = useRouter();
const { setErrorTxt } = useLogin();

localStorage.set('userId', 'zzp');

route.beforeEach(async (to, from, next) => {
  next();
});

getTokenInfo();

async function getTokenInfo() {
  try {
    const loginRes = await urlResquest.loginForToken({ username: 'WYYD', password: 'P@ssword' });
    console.log(loginRes);
    if (loginRes.code === 200) {
      localStorage.set('yongfengToken', loginRes.retObj.token);
      localStorage.set('userId', `yd${loginRes.retObj.user.id}`);
      const checkRes = await urlResquest.checkUser({ encodeLoginName: 'pw2sBCINBJzNShAIMUoWqw==' });
      if (checkRes.code !== 200) {
        throw new Error('用户校验失败，用户不合法');
      } else {
        const res = await urlResquest.getLoginTime({
          code: '8nXGpQIez0cq8OFxSy1o3V5QQtWAuwdWl0Kcu2ntTJ8=',
        });
        console.log(res);
      }
    } else {
      throw new Error('登陆失败');
    }
  } catch (e) {
    console.log('loginerr', e);
    setErrorTxt(e);
    return false;
  }
}
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
