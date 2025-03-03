/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 15:41:05
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-31 16:31:05
 * @FilePath: front_end/src/store/useChunkView.ts
 * @Description: 切片预览、编辑弹窗
 */
export const useChunkView = defineStore('useChunkView', () => {
  const showChunkModel = ref(false);

  return {
    showChunkModel,
  };
});
