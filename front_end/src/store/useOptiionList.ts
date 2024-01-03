/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-02 10:40:18
 * @FilePath: /qanything-open-source/src/store/useOptiionList.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

import urlResquest from '@/services/urlConfig';
import { formatFileSize, resultControl, formatDate } from '@/utils/utils';
import { message } from 'ant-design-vue';

export const useOptiionList = defineStore('optionList', () => {
  const dataSource = ref([]);
  const setDataSource = (array: []) => {
    dataSource.value = array;
  };

  const tmpArr = ref([]); //用来判断是否更新dataSource

  const getDetails = async kb_id => {
    try {
      const res: any = await resultControl(await urlResquest.fileList({ kb_id: kb_id }));
      if (!tmpArr.value?.length) {
        tmpArr.value = res?.details;
      } else if (tmpArr.value.length) {
        const str = JSON.stringify(tmpArr.value);
        const str1 = JSON.stringify(res?.details);
        if (str === str1) {
          console.log('不更新');
          return;
        } else {
          console.log('更新');
          tmpArr.value = res?.details;
        }
      }

      setDataSource([]);
      res?.details.forEach((item: any, index) => {
        dataSource.value.push({
          id: 10000 + index,
          file_id: item?.file_id,
          file_name: item?.file_name,
          status: item?.status,
          bytes: formatFileSize(item?.bytes || 0),
          createtime: formatDate(item?.timestamp),
          errortext: item?.status === 'gray' || item?.status === 'green' ? '' : item?.msg,
        });
      });
    } catch (error) {
      console.log(error);
      message.error(error.msg || '获取知识库详情失败');
    }
  };

  return {
    dataSource,
    setDataSource,
    getDetails,
    tmpArr,
  };
});
