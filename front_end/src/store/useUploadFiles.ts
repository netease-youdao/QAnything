import { IFileListItem } from '@/utils/types';

/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-07-24 14:31:19
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 11:46:27
 * @FilePath: front_end/src/store/useUploadFiles.ts
 * @Description: 上传文件的文件列表管理
 */
export const useUploadFiles = defineStore('useUploadFiles', () => {
  // 上传的文件列表
  const uploadFileList = ref<IFileListItem[]>([]);

  // 初始化（重置）文件列表
  const initUploadFileList = () => {
    uploadFileList.value = [];
  };

  // 本次上传的文件上传列表
  const curUploadFileNum = ref(0);

  const initCurUploadFileNum = () => {
    curUploadFileNum.value = 0;
  };

  return {
    uploadFileList,
    initUploadFileList,
    curUploadFileNum,
    initCurUploadFileNum,
  };
});
