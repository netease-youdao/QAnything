/*
 * @Author: zhangxx03 zhangxx03@rd.netease.com
 * @Date: 2023-02-08 14:13:33
 * @LastEditors: zhangxx03 zhangxx03@rd.netease.com
 * @LastEditTime: 2023-02-22 11:45:26
 * @FilePath: /ai-demo/src/services/axiosInterceptor/interceptors/index.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import autoRetry from './autoRetry';
import cancelRepeat from './cancelRepeat';
import sign from './sign';
import errorToast from './errorToast';
import forceRetry from './forceRetry';
import showLoading from './showLoading';

export default {
  autoRetry,
  cancelRepeat,
  errorToast,
  forceRetry,
  showLoading,
  sign,
  // rdLoginReqToken,
};
