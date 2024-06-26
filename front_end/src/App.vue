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
const { isLogin } = storeToRefs(useLogin());
const { setErrorTxt, setIsLogin } = useLogin();

localStorage.set('userId', 'zzp');

route.beforeEach(async (to, from, next) => {
  console.log('from', from, to);
  if (to.path === '/error') {
    next();
    return false;
  }
  // 先判断是否登陆过
  if (isLogin.value) {
    if (to.query.loginName && to.query.currentTime) {
      next();
    } else if (from.query.loginName && from.query.currentTime) {
      next({
        path: to.path,
        query: { loginName: from.query.loginName, currentTime: from.query.currentTime },
      });
    } else {
      next();
    }
    return false;
  }
  // 如果没有登陆过，并且query里也没有登录信息就跳error
  if (!to.query.loginName || !to.query.currentTime) {
    next('/error');
    return false;
  }

  let loginName = to.query.loginName;
  if (Array.isArray(loginName)) {
    // 如果是字符串数组，则取第一个值
    loginName = loginName[0];
  }
  // 然后进行URL解码
  loginName = decodeURIComponent(loginName).replace(/\s/g, '+');
  console.log('loginName', loginName);

  let currentTime = to.query.currentTime;
  if (Array.isArray(currentTime)) {
    currentTime = currentTime[0];
  }
  currentTime = decodeURIComponent(currentTime).replace(/\s/g, '+');
  console.log('currentTime', currentTime);

  // query有登录信息，调用第三方登录验证
  const boolean = await getTokenInfo(loginName, currentTime, 0);
  console.log('boolean', boolean);
  if (boolean) {
    setIsLogin(true);
    next();
  } else {
    next('/error');
  }
});

// 如果接口401说明token过期，重新获取一下token，state用来限制只重复获取一次token
async function getTokenInfo(loginName, currentTime, state) {
  try {
    if (localStorage.get('yongfengToken') && state < 1) {
      // 获取解密的userName作为传给QAnything底层的user_id
      const res = await urlResquest.getDeCodeString({ code: loginName });
      if (res.code === 401 && state <= 1) {
        return getTokenInfo(loginName, currentTime, state + 1);
      }
      if (res.code === 200) {
        localStorage.set('userId', `yd${res.retObj}`);
      }
      return verifyUser(loginName, currentTime, state);
    } else {
      const loginRes = await urlResquest.loginForToken({ username: 'WYYD', password: 'P@ssword' });
      console.log(loginRes);
      if (loginRes.code === 200) {
        localStorage.set('yongfengToken', loginRes.retObj.token);

        // 获取解密的userName作为传给QAnything底层的user_id
        const res = await urlResquest.getDeCodeString({ code: loginName });
        if (res.code === 401 && state <= 1) {
          return getTokenInfo(loginName, currentTime, state + 1);
        }
        if (res.code === 200) {
          localStorage.set('userId', `yd${res.retObj}`);
        }
        return verifyUser(loginName, currentTime, state);
      } else {
        throw new Error(loginRes?.msg || '登陆失败');
      }
    }
  } catch (e) {
    console.log('loginerr', e);
    setErrorTxt(String(e).replace('Error:', ''));
    return false;
  }
}

async function verifyUser(loginName, currentTime, state) {
  try {
    const checkRes = await urlResquest.checkUser({ encodeLoginName: loginName });
    if (checkRes.code === 401 && state <= 1) {
      return getTokenInfo(loginName, currentTime, state + 1);
    }
    if (checkRes.code !== 200) {
      throw new Error(checkRes?.msg || '用户校验失败，用户不合法');
    } else {
      const res = await urlResquest.getDeCodeString({ code: currentTime });
      if (checkRes.code === 401 && state <= 1) {
        return getTokenInfo(loginName, currentTime, state + 1);
      }
      if (res.code === 200) {
        // 加密串如果没有解析出来会返回空字符串
        if (!res?.retObj) {
          throw new Error('登录超时');
        } else {
          const time: any = new Date(res?.retObj);
          const currentTime: any = new Date('2024-05-13 11:28:11');
          const timeDifference = currentTime - time;
          console.log('time', time, currentTime);
          // 将毫秒转换为分钟
          const minutesDifference = timeDifference / (1000 * 60);
          if (minutesDifference > 30) {
            throw new Error('登录超时');
          } else {
            console.log('未超时');
            return true;
          }
        }
      } else {
        throw new Error(res?.msg || '获取登陆时间失败');
      }
    }
  } catch (e) {
    setErrorTxt(String(e).replace('Error:', ''));
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
