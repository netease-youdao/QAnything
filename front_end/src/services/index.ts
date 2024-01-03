/*
 * @Author: zhangxx zhangxx03@rd.netease.com
 * @Date: 2022-12-05 15:02:44
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-14 14:09:39
 * @FilePath: \ai-demo\src\services\index.ts
 */
import axios from './axiosInterceptor/index';
// import qs from 'qs';

function validateStatus(status: number) {
  return status >= 200 && status < 300;
}
// const transformRequest = [
//   function (data) {
//     console.log('data', data);
//     console.log(qs.stringify(data));
//     return qs.stringify(data);
//   },
// ];

//获取到当前业务线之后设置
export const bondParams = {};

export default {
  get(baseUrl: string, _query = {} as any, option = {} as any) {
    let url = /http/.test(baseUrl)
      ? `${baseUrl}`
      : `${import.meta.env.VITE_APP_SERVER_URL}${baseUrl}`;
    const query = {
      ...bondParams,
      ..._query,
    };
    // Object.keys(query).forEach(key => {
    //   var val = query[key];
    //   url += `${key}=${encodeURIComponent(val)}&`;
    // });
    const { getResponseHeader, ...others } = option;
    const options = {
      method: 'get',
      url,
      mode: 'cors',
      withCredentials: true,
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
    const url = /http/.test(baseUrl) ? baseUrl : `${import.meta.env.VITE_APP_SERVER_URL}${baseUrl}`;
    // const fd = new FormData();
    const { getResponseHeader, ...others } = option; //deepKey,
    // Object.keys(params).forEach(key => {
    //   if (deepKey && key === deepKey && Array.isArray(params[key])) {
    //     params[key].forEach((file: any) => {
    //       fd.append(`${key}`, file);
    //     });
    //     delete option.deepKey;
    //   } else {
    //     fd.append(key, params[key]);
    //   }
    // });
    const options = {
      method: 'post',
      url,
      mode: 'cors',
      withCredentials: true,
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
