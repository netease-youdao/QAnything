/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:49
 * @FilePath: /QAnything/front_end/src/utils/utils.ts
 * @Description:
 */

import { useUser } from '@/store/useUser';
import { IFileListItem } from './types';

export function addWindowsAttr(name, value) {
  window[name] = value;
}
export function getRandomString(strLen = 5) {
  const strCeils = 'abcdefghijklmnopqrstuvwxyz1234567890';
  let str = '';
  for (let i = 0; i < strLen; i += 1) {
    str = `${str}${strCeils[Math.floor(Math.random() * 36)]}`;
  }
  return str;
}
export function clearTimer(timer) {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

export function aDownLoad(url, fileName = '', callback?) {
  let aLink = document.createElement('a');
  aLink.download = fileName;
  aLink.style.display = 'none';
  aLink.href = url;
  document.body.appendChild(aLink);
  aLink.click();
  document.body.removeChild(aLink);
  if (callback) {
    callback();
  }
}

export function isMac() {
  return /macintosh|mac os x/i.test(navigator.userAgent);
}

/**
 * 节流
 * @param {*} fn
 * @param {*} delay
 */
export const throttle = (fn, delay) => {
  var timer,
    last = 0;
  return function () {
    var _this = this;
    var args = arguments;
    var now = +Date.now();
    if (now - last >= delay) {
      clearTimeout(timer);
      fn.apply(_this, args);
      last = now;
    } else {
      clearTimeout(timer);
      timer = setTimeout(function () {
        fn.apply(_this, args);
      }, delay);
    }
  };
};

//格式化上传状态
export const getStatus = (item: IFileListItem) => {
  let str = '';
  switch (item.status) {
    case 'loading':
      str = '上传中';
      break;
    case 'red':
      if (item.errorText) {
        str = item.errorText;
      } else {
        str = '解析失败';
      }
      break;
    case 'gray':
      str = '上传成功待解析';
      break;
    case 'green':
      str = '解析成功';
      break;
    case 'yellow':
      str = '解析失败';
      break;
    default:
      break;
  }
  return str;
};

//对接口的返回值作统一处理
export const resultControl = async res => {
  return new Promise((resolve, reject) => {
    if (res.errorCode === '0' || res.code === 200) {
      resolve(res.result || res.data || res);
    } else if (res.errorCode === '111') {
      const { setUserInfo } = useUser();
      setUserInfo({ token: '' });
    } else {
      reject(res);
    }
  });
};

export const formatFileSize = sizeInBytes => {
  if (sizeInBytes < 0) {
    return '未知';
  } else if (sizeInBytes < 1024) {
    return sizeInBytes + 'B';
  } else if (sizeInBytes < 1024 * 1024) {
    return (sizeInBytes / 1024).toFixed(2) + 'KB';
  } else if (sizeInBytes < 1024 * 1024 * 1024) {
    return (sizeInBytes / (1024 * 1024)).toFixed(2) + 'MB';
  } else {
    return (sizeInBytes / (1024 * 1024 * 1024)).toFixed(2) + 'G';
  }
};

export const formatDate = (timestamp: string, symbol = '-') => {
  if (timestamp) {
    const year = timestamp.slice(0, 4);
    const month = timestamp.slice(4, 6);
    const day = timestamp.slice(6, 8);
    return year + symbol + month + symbol + day;
  } else {
    return '';
  }
};
