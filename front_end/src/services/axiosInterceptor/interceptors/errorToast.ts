import checkResStatus from '@/services/ResConfig';
import { message } from 'ant-design-vue';

function response(res: any) {
  const { config = {} } = res || {};
  // config.errorToast &&
  if (!checkResStatus.isSuccess(res.data.code) && config.errorToast !== false) {
    //res.data.msg
    // message.error('请求失败，请重试');
  }
  return res;
}
function responseError(err: any) {
  message.error('请求失败，请重试');
  return Promise.reject(err);
}
export default {
  response,
  responseError,
};
