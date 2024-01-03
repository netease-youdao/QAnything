function responseError(err = {}, instance: any) {
  const { config } = err as any;
  const { url, retryCount = 3, curRetry = 0, retryDelay } = config as any;

  if (!config.curRetry) {
    config.curRetry = 0;
  }
  if (retryCount > curRetry) {
    config.curRetry += 1;
    const delay = new Promise((resolve: any) => {
      setTimeout(() => {
        resolve();
      }, retryDelay);
    });
    // 重新发起请求
    return delay.then(() => {
      console.log(`重试:${url},第${config.curRetry}次`);
      return instance(config);
    });
  }

  return Promise.reject(err);
}
function response(res: any) {
  return res;
}
export default {
  response,
  responseError,
};
