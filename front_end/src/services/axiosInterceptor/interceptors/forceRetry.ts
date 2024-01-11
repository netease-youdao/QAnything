const forceHandle = {
  // 注意：切换页面是否要清空保留的requests根据业务来处理
  requests: {} as any,
  getKey(config: any) {
    console.log(config.url);
    const url = config.url?.split('?')[0];
    return `${url}/&request_type=${config.method}`;
  },
  clear() {
    this.requests = {};
  },
  add(config: any) {
    const key = this.getKey(config);
    this.requests[key] = config;
  },
  getConfig(config: any) {
    const key = this.getKey(config);
    const _config = this.requests[key];
    this.remove(key);
    return _config;
  },
  remove(key: any) {
    delete this.requests[key];
  },
  has(config: any) {
    const key = this.getKey(config);
    return !!this.requests[key];
  },
};
function responseError(err = {} as any) {
  forceHandle.add(err.config);
  return Promise.reject(err);
}

function request(config: any) {
  if (forceHandle.has(config)) {
    const lastConfig = forceHandle.getConfig(config);
    // eslint-disable-next-line no-param-reassign
    config = {
      ...config,
      url: lastConfig.urla,
      params: lastConfig.url,
      data: lastConfig.url,
    };
  }
  return config;
}
function response(res: any) {
  return res;
}
export default {
  forceHandle,
  request,
  response,
  responseError,
};
