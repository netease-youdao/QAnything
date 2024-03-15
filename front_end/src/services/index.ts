/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
k * @LastEditTime: 2024-01-11 11:33:16
 * @FilePath: /QAnything/front_end/src/services/index.ts
 * @Description:
 */

import axios from './axiosInterceptor/index';

export const apiBase =
  (import.meta.env.VITE_APP_MODE === 'dev' ? '' : import.meta.env.VITE_APP_API_HOST) +
  import.meta.env.VITE_APP_API_PREFIX;

function validateStatus(status: number) {
  return status >= 200 && status < 300;
}

//获取到当前业务线之后设置
export const bondParams = {};

export default {
  get(baseUrl: string, _query = {} as any, option = {} as any) {
    let url = /http/.test(baseUrl) ? `${baseUrl}` : `${apiBase}${baseUrl}`;
    const query = {
      ...bondParams,
      ..._query,
    };

    const { getResponseHeader, ...others } = option;
    const options = {
      method: 'get',
      url,
      mode: 'cors',
      withCredentials: false,
      validateStatus,
      // transformRequest,
      ...others,
      params: query,
    };
    const data = axios.request(options).then(
      res => (getResponseHeader ? res : res.data),
      error => error
    );
    return data;
  },
  post(baseUrl: string, data = {}, option = {} as any) {
    const params = {
      ...bondParams,
      ...data,
    } as any;
    const _url = `${apiBase}${baseUrl}`;
    const url = /http/.test(baseUrl) ? baseUrl : _url;
    const { getResponseHeader, ...others } = option;

    const options = {
      method: 'post',
      url,
      mode: 'cors',
      withCredentials: false,
      validateStatus,
      // transformRequest,
      data: params,
      ...option,
      ...others,
    };
    const resData = axios.request(options).then(
      res => (getResponseHeader ? res : res.data),
      error => Promise.reject(error)
    );
    return resData;
  },
} as any;
