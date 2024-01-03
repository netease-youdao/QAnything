/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:59:58
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-02 11:43:09
 * @FilePath: /qanything-open-source/src/store/useKnowledgeBase.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import { IKnowledgeItem } from '@/utils/types';
import { pageStatus } from '@/utils/enum';
// import { resultControl } from '@/utils/utils';
import message from 'ant-design-vue/es/message';

import urlResquest from '@/services/urlConfig';

export const useKnowledgeBase = defineStore('knowledgeBase', () => {
  // 当前操作的知识库id
  const currentId = ref('');
  const setCurrentId = (id: string) => {
    currentId.value = id;
  };

  //选中的知识库id
  const selectList = ref([]);
  const setSelectList = list => {
    selectList.value = list;
  };

  // 当前操作的知识库名字
  const currentKbName = ref('');
  const setCurrentKbName = (id: string) => {
    currentKbName.value = id;
  };

  //获取到的知识库列表
  const knowledgeBaseList = ref<Array<IKnowledgeItem>>([]);
  const setKnowledgeBaseList = list => {
    knowledgeBaseList.value = list;
  };

  //需要判断是否有知识库 如果没有知识库 展示default内容
  const showDefault = ref(pageStatus.initing);
  const setDefault = str => {
    showDefault.value = str;
  };

  //是否展示删除弹窗
  const showDeleteModal = ref(false);
  const setShowDeleteModal = (flag: boolean) => {
    showDeleteModal.value = flag;
  };

  //获取知识库列表
  const getList = async () => {
    try {
      const res: any = await urlResquest.kbList();
      if (+res.code === 200) {
        setKnowledgeBaseList(res.data);
        if (res?.data?.length > 0) {
          setDefault(pageStatus.normal);

          if (!selectList.value.length) {
            selectList.value.push(res.data[0].kb_id);
          }
        } else {
          setDefault(pageStatus.default);
        }
      }
    } catch (e) {
      message.error(e.msg || '请求失败');
    }
  };

  return {
    currentId,
    setCurrentId,
    knowledgeBaseList,
    setKnowledgeBaseList,
    showDeleteModal,
    setShowDeleteModal,
    showDefault,
    setDefault,
    getList,
    currentKbName,
    setCurrentKbName,
    selectList,
    setSelectList,
  };
});
