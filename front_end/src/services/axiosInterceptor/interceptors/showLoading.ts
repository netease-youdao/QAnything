import { message } from 'ant-design-vue';

const loadingHandle = {
  ids: {} as any,
  addLoading(config: any) {
    config.hideLoading = message.loading({ content: 'loading', duration: 0 });
  },
  addId(config: any) {
    const { loadingId, hideLoading } = config;
    if (!this.ids[loadingId]) {
      this.ids[loadingId] = { loadingNum: 1, hideLoading };
    } else {
      this.ids[loadingId].loadingNum += 1;
      config.hideLoading = this.ids[loadingId].hideLoading;
    }
  },
  canShow(loadingId: any) {
    return !this.ids[loadingId] || this.ids[loadingId].loadingNum === 0;
  },
  removeId(loadingId: any) {
    if (this.ids[loadingId]) {
      this.ids[loadingId].loadingNum -= 1;
      if (this.ids[loadingId].loadingNum === 0) {
        delete this.ids[loadingId];
      }
    }
  },
};
function request(config: any) {
  const { loadingId } = config;
  if (loadingId) {
    if (loadingHandle.canShow(loadingId)) {
      loadingHandle.addLoading(config);
    }
    loadingHandle.addId(config);
  } else {
    loadingHandle.addLoading(config);
  }
  return config;
}
function response(res: any) {
  const { config = {} } = res || {};
  const { hideLoading, loadingId } = config;
  if (hideLoading) {
    if (loadingId) {
      loadingHandle.removeId(loadingId);
      if (loadingHandle.canShow(loadingId)) {
        hideLoading();
      }
    } else {
      hideLoading();
    }
  }
  return res;
}
function responseError(err: any) {
  return Promise.reject(response(err));
}
export default {
  request,
  response,
  responseError,
};
