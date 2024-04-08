/*
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-05 19:12:04
 * @FilePath: /ai-demo/src/store/useOptiionList.ts
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

import urlResquest from '@/services/urlConfig';
import { formatFileSize, resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import moment from 'moment';
const { currentId } = storeToRefs(useKnowledgeBase());

export const useOptiionList = defineStore(
  'option-list',
  () => {
    const dataSource = ref([]);
    const setDataSource = (array: []) => {
      dataSource.value = array;
    };

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
      try {
        if (timer.value) {
          clearTimeout(timer.value);
        }
        const res: any = await resultControl(await urlResquest.fileList({ kbId: currentId.value }));

        setDataSource([]);
        res?.forEach((item: any, index) => {
          dataSource.value.push({
            id: 10000 + index,
            fileId: item?.fileId,
            fileIdName: item?.fileName,
            status: +item?.status,
            bytes: formatFileSize(item?.fileSize),
            createtime: item?.createTime?.split(' ')[0],
            errortext: item?.status == 1 || item?.status == 0 ? '' : item?.remark,
          });
        });

        const flag = res?.some(item => {
          return item.status === '0';
        });
        console.log(flag);
        if (flag) {
          console.log('有解析中的  10后再次请求');
          //有解析中的
          timer.value = setTimeout(() => {
            clearTimeout(timer.value);
            getDetails();
          }, 10000);
        } else {
          console.log('全部解析完成');
        }
      } catch (error) {
        console.log(error);
        message.error(error.msg || '获取知识库详情失败');
      }
    };

    const faqTimer = ref(null);

    const getFaqList = async pageId => {
      try {
        if (faqTimer.value) {
          clearTimeout(faqTimer.value);
        }
        setLoading(true);
        const res: any = await resultControl(
          await urlResquest.faqList({
            kbId: currentId.value,
            page: pageId,
            size: 10,
          })
        );

        setFaqList([]);
        if (!res?.faqList) {
          setTotal(0);
          setLoading(false);
          return;
        }
        setTotal(res.total);
        res?.faqList.forEach(async (item, i) => {
          faqList.value.push({
            id: item?.faq.id,
            faqId: item?.faq.faqId,
            kbId: item?.faq.kbId,
            question: item?.faq.question,
            answer: item?.faq.answer,
            status: +item?.faq.status,
            bytes: `${item?.faq.size}字符`,
            createtime: moment(item?.faq.createTime).format('YYYY-MM-DD'),
            picUrlList: [],
          });
          // 格式化图片为upload支持的结构
          if (item?.picUrlList) {
            await fetchImagesAsFiles(item?.picUrlList)
              .then(files => {
                // 在这里可以使用获取到的File对象数组
                console.log('成功获取图片文件:', files);
                // 进行赋值操作或其他处理
                const imgs = item?.picUrlList.map((img, index) => {
                  return {
                    uid: -index,
                    name: 'image',
                    status: 'done',
                    url: img,
                    originFileObj: files[index],
                  };
                });
                faqList.value[i].picUrlList = imgs;
              })
              .catch(errors => {
                console.error('获取图片文件出错:', errors);
              });
          }
        });

        const flag = res?.faqList.some(item => {
          return +item.faq.status === 0;
        });
        console.log('flag', flag);
        if (flag) {
          console.log('有解析中的  5s后再次请求');
          //有解析中的
          faqTimer.value = setTimeout(() => {
            clearTimeout(faqTimer.value);
            getFaqList(pageId);
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
      setTotal,
      pageNum,
      setPageNum,
      loading,
      setLoading,
    };
  },
  {
    persist: {
      storage: sessionStorage,
    },
  }
);
