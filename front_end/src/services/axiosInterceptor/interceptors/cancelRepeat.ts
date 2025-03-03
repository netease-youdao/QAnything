import axios from 'axios';

const pending = {
  axios: null,
  requests: {} as any,
  clear(curKey: string) {
    if (curKey) {
      delete this.requests[curKey];
    } else {
      this.requests = {};
    }
  },
  setAxios(_axios: any) {
    if (!this.axios) {
      this.axios = _axios;
    }
  },
  has(config: any) {
    const requestKey = this.getKey(config);
    // eslint-disable-next-line no-param-reassign
    config.requestKey = requestKey;
    return !!this.requests[config.requestKey];
  },
  getKey(config: any) {
    return `${config.url}/${JSON.stringify(config.params)}/${JSON.stringify(
      config.data
    )}&request_type=${config.method}`;
  },
  remove(config: any) {
    if (!Object.keys(this.requests).length) return;
    const curKey = config.requestKey;
    if (this.requests[curKey]) {
      this.requests[curKey]();
      this.clear(curKey);
    }
  },
  cancel(config: any, callback: () => {}) {
    const { CancelToken } = axios;
    // eslint-disable-next-line no-new
    // eslint-disable-next-line no-param-reassign
    config.cancelToken = new CancelToken(callback);
  },
  add(config: any) {
    const requestKey = pending.getKey(config);
    // eslint-disable-next-line no-param-reassign
    config.requestKey = requestKey;
    // this.cancel(config, cancel => {
    //   this.requests[config.requestKey] = cancel;
    // });
  },
};

function request(config: any) {
  if (!pending.has(config)) {
    pending.add(config);
  } else {
    // pending.cancel(config, cancel => {
    //   // 取消当次
    //   cancel();
    //   console.log('重复的请求被主动拦截: ---', config.requestKey);
    // });
  }
  return config;
}
function response(res: any) {
  pending.remove(res.config);
  return res;
}
// function error(err) {
//   if (!axios.isCancel(err)) {
//     pending.clear();
//   }
//   return err
// }
export default {
  pending,
  request,
  response,
  // error,
};
