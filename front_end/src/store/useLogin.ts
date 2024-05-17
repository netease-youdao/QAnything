export const useLogin = defineStore('useLogin', () => {
  // 是否验证登录
  const isLogin = ref(false);
  const setIsLogin = value => {
    isLogin.value = value;
  };

  // 登录时间戳 超过30分钟视为过期
  const timeStamp = ref(null);
  const setTimeStamp = value => {
    timeStamp.value = value;
  };

  // 校验登录失败文本
  const errorTxt = ref('登陆失败');
  const setErrorTxt = value => {
    errorTxt.value = value;
  };

  return {
    isLogin,
    setIsLogin,
    timeStamp,
    setTimeStamp,
    errorTxt,
    setErrorTxt,
  };
});
