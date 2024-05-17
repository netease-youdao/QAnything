/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:40:28
 * @FilePath: /QAnything/front_end/src/services/urlConfig.ts
 * @Description:
 */

enum EUrlType {
  POST = 'post',
  GET = 'get',
}
interface IUrlValueConfig {
  type: EUrlType;
  url: string;
  showLoading?: boolean;
  loadingId?: string;
  // errorToast?: boolean;//默认开启
  cancelRepeat?: boolean;
  sign?: boolean; // 是否开启签名
  param?: any;
  [key: string]: any;
}
interface IUrlConfig {
  [key: string]: IUrlValueConfig;
}
import services from './loginService';

const baseUrl = 'https://www.yongfengai.com/appspace/v1';

//ajax请求接口
const urlConfig: IUrlConfig = {
  //登录
  loginForToken: {
    type: EUrlType.POST,
    url: baseUrl + '/loginForToken',
  },
  //检查用户是否合法
  checkUser: {
    type: EUrlType.POST,
    url: baseUrl + '/system/user/checkUserByEncryptCode',
  },
  //获取登录时间戳
  getLoginTime: {
    type: EUrlType.POST,
    url: baseUrl + '/encode/getDeCodeString',
  },
};
const urlResquest: any = {};
Object.keys(urlConfig).forEach(key => {
  urlResquest[key] = (params: any, option: any = {}) => {
    const { type, url, param, ...other } = urlConfig[key];
    return services[type](url, { ...param, ...params }, { ...other, ...option });
  };
});
export default urlResquest;
