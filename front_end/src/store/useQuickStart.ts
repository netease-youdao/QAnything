/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-22 18:18:48
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-23 09:53:40
 * @FilePath: front_end/src/store/useQuickStart.ts
 * @Description: 这是默认设置,可以在设置》工具》File Description中进行配置
 */
interface IHistoryList {
  historyId: number;
  kbId: string;
  title: string;
}
export const useQuickStart = defineStore('useQuickStart', () => {
  const history = ref<IHistoryList[]>([]);
});
