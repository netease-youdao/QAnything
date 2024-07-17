/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:35
 * @FilePath: /QAnything/front_end/src/utils/types.ts
 * @Description:
 */

export interface IKnowledgeItem {
  kb_id: string;
  kb_name: string;
  isFaq?: boolean;
  createTime?: any;
  edit?: boolean;
}

export interface IDataSourceItem {
  dataSource?: string; //数据来源
  detailDataSource?: string; //详细来源信息
  file_name: string | null; //文件名
  content: string | null; //内容
  score: number | null; //评分
  file_id: string | null;
  showDetailDataSource?: boolean; //是否展示详细来源信息
}

export interface IChatItem {
  type: string; //区别用户提问 和ai回复
  question?: string; //问题
  answer?: string; //问题 | 回复内容
  like?: boolean; //点赞
  unlike?: boolean; //点踩
  copied?: boolean; //点拷贝置为true 提示拷贝成功 然后置为false  重置原因:点击拷贝后添加颜色提示拷贝过了 1s后置为普通颜色
  onlySearch?: boolean; // 只检索知识库来源不回答

  showTools?: boolean; //当期问答是否结束 结束展示复制等小工具和取消闪烁
  source?: Array<IDataSourceItem>;
}

//url解析状态（前端展示）
export type inputStatus = 'default' | 'inputing' | 'parsing' | 'success' | 'defeat' | 'hover';

//url类型约束
export interface IUrlListItem {
  status: inputStatus;
  text: string;
  percent: number;
  borderRadius?: string;
}

//上传文件
export interface IFileListItem {
  file?: File;
  file_name: string;
  status: string;
  file_id: string;
  percent?: number;
  errorText?: string;
}

// 模型设置
type ICapabilities = {
  /* 是否联网搜索 */
  onlineSearch: boolean;
  /* 是否混合搜索 */
  mixedSearch: boolean;
  /* 是否仅检索 */
  onlySearch: boolean;
};
export interface IChatSetting {
  /* 模型类型，不用传 */
  modelType: 'openAI' | 'ollama' | 'custom';
  /* 自定义模型id，如果不是自定义就没有，不用传 */
  customId?: number;
  /* 自定义的模型名称，只有自定义时候用 */
  modelName?: string;
  /* 秘钥，openAI用 */
  apiKey?: string;
  /* api路径 */
  apiBase: string;
  /* 模型名称 */
  apiModelName: string;
  /* 上下文token数量 */
  apiContextLength: number;
  /* 上下文的消息数量上限条数，不用传 */
  context: number;
  /*  */
  maxToken: number;
  /* 联想与发散 0~1 */
  temperature: number;
  /* top_P 0~1 */
  top_P: number;
  /* 模型能力 */
  capabilities: ICapabilities;
}
