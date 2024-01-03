/*
 * @Author: zhangxx03 zhangxx03@rd.netease.com
 * @Date: 2023-02-08 14:13:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-02 11:23:52
 * @FilePath: /qanything-open-source/src/services/urlConfig.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
// http://confluence.inner.youdao.com/pages/viewpage.action?pageId=103592897#id-%E7%A1%AC%E4%BB%B6%E6%96%87%E4%BB%B6%E5%90%8C%E6%AD%A5%EF%BC%88%E4%BA%91%E7%9B%98%EF%BC%89--%E6%8E%A5%E5%8F%A3%E6%96%87%E6%A1%A3%EF%BC%88%E5%90%8E%E5%8F%B0%EF%BC%89-4.%E7%B3%BB%E7%BB%9F%E5%BC%82%E6%AD%A5%E5%88%B6%E4%BD%9C%E5%AD%97%E5%B9%95%E5%B9%B6%E4%B8%8A%E4%BC%A0%E5%88%B0%E5%8D%95%E6%9B%B2
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
import services from '.';

export const userId = 'zhujietest4';

//ajax请求接口
const urlConfig: IUrlConfig = {
  checkLogin: {
    type: EUrlType.GET,
    url: '/checkLogin.s',
  },
  getLoginInfo: {
    type: EUrlType.POST,
    url: '/j_spring_security_check',
  },
  // 获取知识库列表
  kbList: {
    type: EUrlType.POST,
    url: '/local_doc_qa/list_knowledge_base',
    showLoading: true,
    param: {
      user_id: userId,
    },
  },
  // 新建知识库
  createKb: {
    type: EUrlType.POST,
    url: '/local_doc_qa/new_knowledge_base',
    showLoading: true,
    param: {
      user_id: userId,
    },
  },
  // 上传文件
  uploadFile: {
    type: EUrlType.POST,
    url: '/local_doc_qa/upload_files',
    param: {
      user_id: userId,
    },
  },
  // 删除知识库
  deleteKB: {
    type: EUrlType.POST,
    url: '/local_doc_qa/delete_knowledge_base',
    param: {
      user_id: userId,
    },
  },
  // 删除文件
  deleteFile: {
    type: EUrlType.POST,
    url: '/local_doc_qa/delete_files',
    showLoading: true,
    param: {
      user_id: userId,
      kb_id: '',
      file_ids: [],
    },
  },
  // 上传网页文件
  uploadUrl: {
    type: EUrlType.POST,
    url: '/local_doc_qa/upload_weblink',
    param: {
      user_id: userId,
    },
  },
  kbConfig: {
    type: EUrlType.POST,
    url: '/local_doc_qa/rename_knowledge_base',
    showLoading: true,
    param: {
      user_id: userId,
      kb_id: '',
      new_kb_name: '',
    },
  },
  //获取知识库已上传文件状态
  fileList: {
    type: EUrlType.POST,
    url: '/local_doc_qa/list_files',
    param: {
      user_id: userId,
      kb_id: '',
    },
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
