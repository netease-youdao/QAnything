/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:18
 * @FilePath: /QAnything/front_end/src/utils/enum.ts
 * @Description:
 */
/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:17
 * @FilePath: /QAnything/front_end/src/utils/enum.ts
 * @Description:
 */

//文档状态说明 （接口返回值）
export enum fileStatus {
  'success', //上传完成
  'parsed', //解析完成
  'error', //解析失败
  'filebig', //上传文件大小超过限制
  'loading', //上传中
}

export enum pageStatus {
  'initing', //初始化 还不知道具体展示什么内容
  'default', //没有知识库 展示默认上传页面
  'normal', //有知识库 展示知识库列表 显示聊天界面
  'optionlist', //管理上传文件页面
}
