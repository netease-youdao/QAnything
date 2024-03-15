/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:44:53
 * @FilePath: /QAnything/front_end/src/services/axiosInterceptor/interceptors/sign.ts
 * @Description:
 */

import checkResStatus from '@/services/ResConfig';
import CryptoJS from 'crypto-js';

const signSecretKey = 'r*bWYmKw0Z@$1^fEk3xIxKqb!3HMTyI';
// const secretKey = '';
// const secretIv = '';
// const algorithm = '';
/**
 * md5加密
 * @param {String} str md5
 * @param {*} decode
 */
function md5(str) {
  return CryptoJS.MD5(str).toString();
}

/**
 * 数据解密
 * @param {*} data
 */
// function decodeData(data) {
//   if (!data) {
//     return null;
//   }
//   const key = Buffer.alloc(16, md5(secretKey)); // 16位密钥 默认 'utf-8'
//   const iv = Buffer.alloc(16, md5(secretIv)); // 16位向量
//   const decipher = crypto.createDecipheriv(algorithm, key, iv);
//   let decrypted = decipher.update(data, 'base64', 'utf-8'); // 原数据 'base64' 解密后数据 ’utf-8‘
//   decrypted += decipher.final('utf-8');
//   return decrypted;
// }
/**
 * 生成签名
 * @param {*} data
 */
function genSign(data = {}) {
  const keySortArr = Object.keys(data)
    .sort()
    .filter(v => data[v] !== undefined && `${data[v]}`.length !== 0);
  const sign = `${keySortArr.map(k => `${k}=${data[k]}`).join('&')}&key=${signSecretKey}`;
  return md5(sign);
}

/**
 * 获取数据对象
 * @param {*} config
 */
function getData(config = {}) {
  const { method, data = {}, params = {} } = config as any;
  switch (method.toLowerCase()) {
    case 'post':
      return data;
    case 'get':
      return params;
    default:
      return params;
  }
}
function response(response) {
  const config = response.config;
  const { data } = response;
  if (data && checkResStatus.isSuccess(data.code)) {
    if (config.decode) {
      // const d = decodeData(data.data);
      data.data = JSON.parse(JSON.parse(data.data));
    }
  } else {
    // if (whiteCodeList.indexOf(data.code) === -1) {
    //   handlerError(data && data.reason);
    // }
  }
  return response;
}

function request(axiosConfig) {
  if (axiosConfig.sign) {
    const signData = getData(axiosConfig);
    signData.sign = genSign(signData);
  }
  return axiosConfig;
}
export default {
  response,
  request,
};
