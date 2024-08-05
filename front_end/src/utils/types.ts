/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 16:36:28
 * @FilePath: front_end/src/utils/types.ts
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
  type: 'ai' | 'user'; //区别用户提问 和ai回复
  question?: string; //问题
  answer?: string; //问题 | 回复内容
  like?: boolean; //点赞
  unlike?: boolean; //点踩
  copied?: boolean; //点拷贝置为true 提示拷贝成功 然后置为false  重置原因:点击拷贝后添加颜色提示拷贝过了 1s后置为普通颜色
  onlySearch?: boolean; // 只检索知识库来源不回答

  showTools?: boolean; //当期问答是否结束 结束展示复制等小工具和取消闪烁
  source?: Array<IDataSourceItem>; // 数据来源

  picList?: any; // 不知道是啥，用到了，不敢删
  qaId?: any; // 同上

  itemInfo?: IChatItemInfo; // 当前对话相关信息 token time chatSetting
}

// 历史记录
export interface IHistoryList {
  historyId: number;
  title: string;
  kbIds?: string[];
}

// 对话的耗时信息
export interface ITimeInfo {
  preprocess: number;
  condense_q_chain: number;
  retriever_search: number;
  web_search: number;
  rerank: number;
  reprocess: number;
  llm_first_return: number;
  first_return: number; // 前7个加起来。外层显示
  llm_completed: number;
  chat_completed: number; // 后俩加起来。外层显示
}

// 对话的耗token信息
export interface ITokenInfo {
  total_tokens: number; // 外层显示
  prompt_tokens: number; // 外层显示
  completion_tokens: number; // 外层显示
  tokens_per_second: number;
}

// 对话的信息：耗token、耗时、当时的模型信息、当时聊天的日期等
export interface IChatItemInfo {
  timeInfo: ITimeInfo; // 耗时相关
  tokenInfo: ITokenInfo; // token相关
  settingInfo: IChatSetting; // 模型配置相关
  dateInfo: number; // 当时聊天的日期，时间戳
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
  text?: string;
  order?: number;
}

// 模型设置
type ICapabilities = {
  /* 是否联网搜索 */
  networkSearch: boolean;
  /* 是否混合搜索 */
  mixedSearch: boolean;
  /* 是否仅检索 */
  onlySearch: boolean;
  /* 是否增强检索 */
  rerank: boolean;
};

export interface IChatSetting {
  /* 模型类型，string为自定义名称，不用传 */
  modelType: 'openAI' | 'ollama' | '自定义模型配置' | string;
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
  /* 返回的最大token */
  maxToken: number;
  /* 切片的token数 */
  chunkSize: number;
  /* 联想与发散 0~1 */
  temperature: number;
  /* top_P 0~1 */
  top_P: number;
  /* 模型能力 */
  capabilities: ICapabilities;
  /* 是否开启（只有一个） */
  active: boolean;
}
