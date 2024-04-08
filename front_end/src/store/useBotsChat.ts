/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2023-12-27 19:21:14
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-29 11:05:53
 * @FilePath: /ai-demo/src/store/useChat.ts
 * @Description:
 */
import { IChatItem } from '@/utils/types';

export const useBotsChat = defineStore({
  id: 'useBotsChat',
  state: () => ({
    QA_List: [],
    showModal: false,
  }),
  actions: {
    clearQAList() {
      this.QA_List = [];
    },
    setQaList(newQaList: Array<IChatItem>) {
      this.QA_List = newQaList;
    },
    // 处理信息来源
    async handleSource(sources) {
      const enrichedSources = await Promise.all(
        sources.map(async source => {
          if (!source.pdf_source_info) {
            console.log('handleSource', source);
            return source;
          }
          const response = await fetch(source.pdf_source_info.chunks_nos_url);
          const chunksInfo = await response.json();
          const { chunks, pageSizes } = this.formatChunks(chunksInfo);
          return { ...source, chunks, pageSizes };
        })
      );

      this.QA_List[this.QA_List.length - 1].source = enrichedSources;
    },
    // 将接口的数据格式化，用于渲染chunks高亮等部分
    formatChunks(chunks) {
      console.log('formatChunks', chunks);
      // pagesInfo为二级数组，第一级代表每一页，第二级代表一个个chunk
      let sizeArr = [];
      let pagesInfoArr = [];
      chunks.forEach(chunk => {
        if (chunk.chunk_type === 'normal') {
          chunk.locations.forEach(item => {
            if (!sizeArr[item.page_id]) {
              sizeArr[+item.page_id] = {
                page_w: item.page_w,
                page_h: item.page_h,
              };
            }
            if (!pagesInfoArr[item.page_id]) {
              pagesInfoArr[item.page_id] = [];
            }
            pagesInfoArr[item.page_id].push({
              chunkId: chunk.chunk_id,
              lines_box: item.lines,
              bbox: item.bbox,
            });
          });
        } else {
          chunk.locations.forEach(item => {
            if (!sizeArr[item.page_id]) {
              sizeArr[+item.page_id] = {
                page_w: item.page_w,
                page_h: item.page_h,
              };
            }
          });
        }
      });
      return {
        chunks: [...pagesInfoArr],
        pageSizes: [...sizeArr],
      };
    },
  },
});
