import service from './index';

const { VITE_APP_ACCOUNT_URL } = import.meta.env;
const simulationImei = new Date().getTime();

// 展示登陆二维码后，轮询该接口获取登陆状态
async function loginstat(params: any) {
  const baseUrl = `${VITE_APP_ACCOUNT_URL}/loginstat`;
  const bondParams = {
    ...params,
    app: 'web',
    from: 'yunting',
    imei: simulationImei, // 选填
    show: 1,
    newjson: 1,
  };
  if (window.location.protocol.indexOf('https') > -1) {
    bondParams.samesite = true;
  }

  return service.get(baseUrl, bondParams);
}

async function accountInfo(params?: any) {
  const baseUrl = `${VITE_APP_ACCOUNT_URL}/query/accountinfo`;
  return service.get(baseUrl, params, { errorToast: false });
}
//退出登录
async function logout(params?: any) {
  const baseUrl = `${VITE_APP_ACCOUNT_URL}/se/reset`;
  return service.get(baseUrl, params);
}

//登录
async function login(params?: any) {
  const baseUrl = `${VITE_APP_ACCOUNT_URL}/corp/silent_auth`;
  return service.get(baseUrl, params);
}
export default {
  login,
  loginstat,
  accountInfo,
  logout,
};
