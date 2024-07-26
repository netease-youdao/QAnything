/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-26 15:41:05
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-07-26 18:10:20
 * @FilePath: front_end/src/store/useChunkView.ts
 * @Description: 切片预览、编辑弹窗
 */
export const useChunkView = defineStore('useChunkView', () => {
  const showChunkModel = ref(false);

  // 当前chunks属于的知识库id
  const chunkKbId = ref('');

  // 当前chunks是哪个file的chunks
  const chunkFileId = ref('');

  return {
    showChunkModel,
    chunkKbId,
    chunkFileId,
  };
});
