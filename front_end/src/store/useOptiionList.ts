/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-06 10:25:59
 * @FilePath: front_end/src/store/useOptiionList.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

import urlResquest from '@/services/urlConfig';
import { formatDate, formatFileSize, resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';

const { currentId } = storeToRefs(useKnowledgeBase());

type Status = 'green' | 'yellow' | 'red' | 'gray';

interface IDataSource {
  id: number;
  bytes: number | string;
  contentLength: number;
  fileId: string;
  fileIdName: string;
  status: Status;
  createtime: string;
  remark: { [key: string]: string } | string;
}

export const useOptiionList = defineStore(
  'option-list',
  () => {
    const dataSource = ref<IDataSource[]>([]);
    const setDataSource = (array: []) => {
      dataSource.value = array;
    };

    const totalStatus = ref({
      green: 0,
      gray: 0,
      yellow: 0,
      red: 0,
    });

    // 知识库文件总数
    const kbTotal = ref(0);
    const setKbTotal = value => {
      kbTotal.value = value;
    };

    // 知识库页号
    const kbPageNum = ref(1);
    const setKbPageNum = value => {
      kbPageNum.value = value;
    };

    // 知识库一页几个
    const kbPageSize = ref(10);

    const faqList = ref([]);
    const setFaqList = (array: []) => {
      faqList.value = array;
    };

    const total = ref(0);
    const setTotal = value => {
      total.value = value;
    };

    // faq table当前页号
    const pageNum = ref(1);
    const setPageNum = value => {
      pageNum.value = value;
    };

    // faq一页几个
    const pageSize = ref(10);

    // faq table loading
    const loading = ref(false);
    const setLoading = value => {
      loading.value = value;
    };

    const faqType = ref('upload'); // upload: 上传 edit: 编辑
    const setFaqType = type => {
      faqType.value = type;
    };

    // 当前正在编辑的问答
    const editQaSet: any = ref(null);
    const setEditQaSet = value => {
      editQaSet.value = value;
    };

    const editModalVisible = ref(false);
    const setEditModalVisible = value => {
      editModalVisible.value = value;
    };

    const timer = ref(null);

    const getDetails = async () => {
      // try {
      if (timer.value) {
        clearTimeout(timer.value);
      }
      const res: any = await resultControl(
        // 接口的page_id为页码，page_limit为一页几个
        await urlResquest.fileList({
          kb_id: currentId.value,
          page_id: kbPageNum.value,
          page_limit: kbPageSize.value,
        })
      );

      // 初始化状态计数
      Object.keys(totalStatus.value).forEach(key => {
        totalStatus.value[key] = 0;
      });

      // 更新状态计数
      Object.assign(totalStatus.value, res.status_count);

      setDataSource([]);

      // 设置一共几个文件
      setKbTotal(res.total);

      // 格式化success
      const computedRemark = (msg: string = '', status: string = 'green') => {
        if (status !== 'green') return msg;
        // stringify转不了, 只能toString()
        return JSON.parse(msg.toString());
      };

      res?.details.forEach((item: any, index) => {
        dataSource.value.push({
          id: 10000 + index,
          fileId: item?.file_id,
          fileIdName: item?.file_name,
          status: item?.status,
          bytes: formatFileSize(item?.bytes || 0),
          contentLength: item?.content_length,
          createtime: formatDate(item?.timestamp),
          remark: item?.status === 'gray' ? '' : computedRemark(item?.msg, item?.status),
        });
      });

      const flag = res?.details.some(item => {
        return item.status === 'gray' || item.status === 'yellow';
      });
      if (flag) {
        //有解析中的
        timer.value = setTimeout(() => {
          clearTimeout(timer.value);
          getDetails();
        }, 5000);
      } else {
        getProgressDetails();
      }
    };

    // 进度条
    const getProgressDetails = () => {
      let timer = null;
      timer = setInterval(async () => {
        const res: any = await resultControl(
          // 接口的page_id为页码，page_limit为一页几个
          await urlResquest.fileList({
            kb_id: currentId.value,
            page_id: 1,
            page_limit: kbPageSize.value,
          })
        );
        // 初始化状态计数
        Object.keys(totalStatus.value).forEach(key => {
          totalStatus.value[key] = 0;
        });

        // 更新状态计数
        Object.assign(totalStatus.value, res.status_count);

        // 设置一共几个文件
        setKbTotal(res.total);

        if (totalStatus.value.gray === 0 && totalStatus.value.yellow === 0) {
          clearInterval(timer);
        }
      }, 5000);
    };

    const faqTimer = ref(null);

    const getFaqList = async () => {
      try {
        if (faqTimer.value) {
          clearTimeout(faqTimer.value);
        }
        setLoading(true);
        const res: any = await resultControl(
          await urlResquest.fileList({
            kb_id: currentId.value + '_FAQ',
            page_id: pageNum.value,
            page_limit: pageSize.value,
          })
        );

        setFaqList([]);
        if (!res?.details) {
          setTotal(0);
          setLoading(false);
          return;
        } else {
          setTotal(res.total);
        }
        for (const item of res?.details) {
          const i = res?.details.indexOf(item);
          faqList.value.push({
            id: 10000 + i,
            faqId: item?.file_id,
            question: item?.question,
            answer: item?.answer,
            status: item?.status,
            bytes: `${item?.content_length}字符`,
            createtime: formatDate(item?.timestamp),
            picUrlList: [],
          });
          // 格式化图片为upload支持的结构
          if (item?.picUrlList) {
            await fetchImagesAsFiles(item?.picUrlList)
              .then(files => {
                // 在这里可以使用获取到的File对象数组
                console.log('成功获取图片文件:', files);
                // 进行赋值操作或其他处理
                faqList.value[i].picUrlList = item?.picUrlList.map((img, index) => {
                  return {
                    uid: -index,
                    name: 'image',
                    status: 'done',
                    url: img,
                    originFileObj: files[index],
                  };
                });
              })
              .catch(errors => {
                console.error('获取图片文件出错:', errors);
              });
          }
        }

        const flag = res?.details.some(item => {
          return item.status === 'gray' || item.status === 'yellow';
        });
        if (flag) {
          console.log('有解析中的  5s后再次请求');
          //有解析中的
          faqTimer.value = setTimeout(() => {
            clearTimeout(faqTimer.value);
            getFaqList();
          }, 5000);
        } else {
          console.log('全部解析完成');
        }
      } catch (error) {
        console.log(error);
        message.error(error.msg || '获取faq列表失败');
      }
      setLoading(false);
    };

    function fetchImagesAsFiles(urls) {
      const promises = urls.map(url => {
        return new Promise((resolve, reject) => {
          fetch(url)
            .then(response => response.blob())
            .then(blob => {
              const filename = url.substring(url.lastIndexOf('/') + 1);
              const file = new File([blob], filename, { type: blob.type });
              resolve(file);
            })
            .catch(error => {
              console.error('Error fetching the image:', error);
              reject(error);
            });
        });
      });

      return Promise.all(promises);
    }

    return {
      dataSource,
      setDataSource,
      faqList,
      setFaqList,
      getDetails,
      timer,
      editQaSet,
      setEditQaSet,
      editModalVisible,
      setEditModalVisible,
      faqTimer,
      getFaqList,
      faqType,
      setFaqType,
      total,
      pageSize,
      setTotal,
      pageNum,
      setPageNum,
      loading,
      setLoading,
      totalStatus,
      kbTotal,
      kbPageSize,
      kbPageNum,
      setKbPageNum,
    };
  },
  {
    persist: {
      storage: sessionStorage,
    },
  }
);
