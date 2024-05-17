<template>
  <Teleport to="body">
    <a-modal
      v-model:open="editModalVisible"
      :title="home.inputQa"
      centered
      :destroyOnClose="true"
      width="480px"
      wrap-class-name="edit-qa-set-modal"
      :footer="null"
    >
      <div class="edit-qa-set-comp">
        <div class="tabs">
          <div
            :class="['tab-item', tabIndex === item.value ? 'tab-item-active' : '']"
            v-for="item in tabList"
            :key="item.name"
            @click="tabClick(item.value)"
          >
            {{ item.name }}
          </div>
        </div>
        <a-form
          v-if="tabIndex === 0"
          :model="formState"
          name="edit_qa_set"
          class="edit-qa-set-form"
          @finish="onFinish"
        >
          <a-form-item name="question" :rules="[{ required: true, message: home.plsInputQ }]">
            <a-input
              v-model:value="formState.question"
              :placeholder="home.setQPlaceholder"
              show-count
              :maxlength="100"
              allow-clear
            />
          </a-form-item>
          <a-form-item name="answer" :rules="[{ required: true, message: home.plsInputA }]">
            <a-textarea
              v-model:value="formState.answer"
              :placeholder="home.setAPlaceholder"
              show-count
              :maxlength="1000"
              :auto-size="{ minRows: 5, maxRows: 5 }"
            />
          </a-form-item>
          <!-- <a-form-item name="imgs">
            <div class="upload-content">
              <a-upload
                v-model:fileList="formState.imgs"
                list-type="picture-card"
                accept=".jpg,.jpeg,.png"
                :before-upload="beforeUpload"
                @preview="handlePreview"
              >
                <div v-if="formState.imgs.length < 3">
                  <PlusOutlined />
                  <div style="margin-top: 8px">{{ home.uploadPictures }}</div>
                </div>
              </a-upload>
              <span class="upload-max">{{ home.uploadMax }}</span>
            </div>
          </a-form-item> -->
          <a-form-item>
            <div class="footer">
              <a-button class="cancel-btn" @click="setEditModalVisible(false)">
                {{ common.cancel }}
              </a-button>
              <a-button :loading="loading" type="primary" html-type="submit" class="login-form-btn">
                {{ common.confirm2 }}
              </a-button>
            </div>
          </a-form-item>
        </a-form>
        <div v-else class="upload-excel">
          <div class="demo-download">
            <a
              href="https://download.ydstatic.com/ead/QAnything_QA_模板.xlsx"
              download="QAnything_QA_模板.xlsx"
            >
              {{ home.downTemp }}
            </a>
            <span>{{ home.uploadXlsxDesc }}</span>
          </div>
          <div class="box">
            <div v-if="!showUploadList" class="before-upload-box">
              <input
                class="hide input"
                type="file"
                :accept="'.xlsx'"
                @change="fileChange"
                @click="e => ((e.target as HTMLInputElement).value = '')"
              />
              <div class="before-upload">
                <div class="upload-text-box">
                  <SvgIcon name="upload" />
                  <p v-if="language === 'zh'">
                    <span class="upload-text"
                      >{{ common.dragUrl }}<span class="blue">{{ common.click }}</span></span
                    >
                  </p>
                  <p v-else>
                    <span class="upload-text"
                      ><span class="blue">{{ common.click }}&nbsp;</span>{{ common.dragUrl }}</span
                    >
                  </p>
                </div>
              </div>
            </div>
            <div
              v-else
              :class="['upload-box', 'upload-list', `upload-${uploadFileList[0].status}`]"
            >
              <UploadList>
                <template #default>
                  <ul class="list">
                    <li v-for="(item, index) in uploadFileList" :key="index">
                      <span class="name">{{ item.file_name }}</span>
                      <div class="status-box">
                        <SvgIcon v-if="item.status != 'loading'" :name="item.status" />
                        <img
                          v-else
                          class="loading"
                          src="../assets/home/icon-loading.png"
                          alt="loading"
                        />
                        <span class="status">{{ item.errorText }}</span>
                      </div>
                    </li>
                  </ul>
                </template>
              </UploadList>
              <div class="overly">
                <input
                  class="hide input"
                  type="file"
                  :accept="'.xlsx'"
                  @change="fileChange"
                  @click="e => ((e.target as HTMLInputElement).value = '')"
                />
                <div class="overly-content">
                  <SvgIcon name="upload-white" />
                  <span>重新上传</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <a-modal
          :open="previewVisible"
          :title="previewTitle"
          :footer="null"
          @cancel="handleCancel"
          width="auto"
          centered
        >
          <img
            alt="example"
            style="max-width: 100%; max-height: 80vh; height: auto; width: auto; margin: 0 auto"
            :src="previewImage"
          />
        </a-modal>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { apiBase } from '@/services';
import { useOptiionList } from '@/store/useOptiionList';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';
// import { PlusOutlined } from '@ant-design/icons-vue';
import { fileStatus } from '@/utils/enum';
import { useLanguage } from '@/store/useLanguage';
import { IFileListItem } from '@/utils/types';
import localStorage from '@/utils/localStorage';

const { setEditModalVisible, setEditQaSet, getFaqList } = useOptiionList();
const { editModalVisible, editQaSet, faqType } = storeToRefs(useOptiionList());
const { currentId } = storeToRefs(useKnowledgeBase());
const home = getLanguage().home;
const common = getLanguage().common;
const { language } = storeToRefs(useLanguage());

interface FormState {
  question: string;
  answer: string;
  imgs: Array<any>;
}

const loading = ref(false);
const formState = reactive<FormState>({
  question: '',
  answer: '',
  imgs: [],
});

const tabIndex = ref(0);
const previewVisible = ref(false);
const previewImage = ref('');
const previewTitle = ref('');

const tabList = [
  {
    name: home.enterQa,
    value: 0,
  },
  {
    name: home.batchUpload,
    value: 1,
  },
];

//是否显示上传文件列表 默认不显示
const showUploadList = ref(false);
const uploadFileList = ref([]); // 本次上传文件列表

watch(
  () => editModalVisible.value,
  () => {
    if (editModalVisible.value) {
      tabIndex.value = 0;
      if (editQaSet.value) {
        formState.question = editQaSet.value?.question;
        formState.answer = editQaSet.value?.answer;
        formState.imgs = editQaSet.value?.picUrlList;
      }
    } else {
      setEditQaSet(null);
      clearFormState();
      showUploadList.value = false;
      uploadFileList.value = [];
    }
  },
  { immediate: true }
);

function clearFormState() {
  formState.question = '';
  formState.answer = '';
  formState.imgs = [];
}

const delFaq = async () => {
  try {
    await resultControl(
      await urlResquest.deleteFile({
        kb_id: `${currentId.value}_FAQ`,
        file_ids: [editQaSet.value?.faqId],
      })
    );
  } catch (e) {
    message.error(e.msg || '删除失败');
  }
};

const onFinish = async (values: any) => {
  console.log('Success:', values);
  loading.value = true;
  // 编辑接口比上传多两个参数
  if (faqType.value === 'edit') {
    await delFaq();
  }
  try {
    const faqs = [
      {
        question: values.question,
        answer: values.answer,
        nos_key: null,
      },
    ];
    const res: any = await resultControl(
      await urlResquest.uploadFaqs({ kb_id: `${currentId.value}_FAQ`, faqs: faqs })
    );
    console.log(res);
    message.success('上传成功');
  } catch (e) {
    message.error(e.msg || '获取Bot信息失败');
  }
  getFaqList();
  loading.value = false;
  setEditModalVisible(false);
};

// const beforeUpload = file => {
//   const isJPG =
//     file.type === 'image/jpeg' || file.type === 'image/jpg' || file.type === 'image/png';
//   console.log('beforeUpload', isJPG, file);
//   if (!isJPG) {
//     message.error('只能上传 JPG、JPEG 和 PNG 格式的图片文件');
//     return Upload.LIST_IGNORE;
//   }
//   formState.imgs = [...formState.imgs, file];
//   return false;
// };

const tabClick = value => {
  tabIndex.value = value;
};

//上传前校验
const beforeFileUpload = async (file, index) => {
  console.log('file', file);
  return new Promise((resolve, reject) => {
    if (file.name.split('.').pop().toLowerCase() === 'xlsx') {
      uploadFileList.value.push({
        file_name: file.name,
        file: file,
        status: 'loading',
        errorText: common.uploading,
        file_id: '',
        order: uploadFileList.value.length,
      });
      resolve(index);
    } else {
      reject(file.name);
    }
  });
};

//input上传
const fileChange = e => {
  uploadFileList.value = [];
  const files = e.target.files;
  Array.from(files).forEach(async (file: any, index) => {
    try {
      await beforeFileUpload(file, index);
    } catch (e) {
      message.error(`${e}的文件格式不符`);
    }
  });
  setTimeout(() => {
    uploadFileList.value.length && uplolad();
  });
};

const uplolad = async () => {
  const list = [];
  showUploadList.value = true;
  console.log('uploadFileList', uploadFileList.value);
  uploadFileList.value.forEach((file: IFileListItem) => {
    if (file.status === 'loading') {
      list.push(file);
    }
  });
  const formData = new FormData();
  for (let i = 0; i < list.length; i++) {
    formData.append('files', list[i]?.file);
  }
  formData.append('user_id', localStorage.get('userId'));
  formData.append('kb_id', `${currentId.value}_FAQ`);

  fetch(apiBase + '/local_doc_qa/upload_faqs', {
    method: 'POST',
    body: formData,
  })
    .then(response => {
      console.log(response);

      if (response.ok) {
        return response.json(); // 将响应解析为 JSON
      } else {
        throw new Error('上传失败');
      }
    })
    .then(data => {
      console.log(data);
      // 在此处对接口返回的数据进行处理
      if (data.code === 200) {
        list.forEach((item, index) => {
          let status = data.data[index].status;
          if (status == 'green' || status == 'gray') {
            status = 'success';
          } else {
            status = 'error';
          }
          uploadFileList.value[item.order].status = status;
          uploadFileList.value[item.order].errorText = common.upSucceeded;
        });
        setEditModalVisible(false);
        message.success('上传成功');
      } else {
        message.error(data.msg || '出错了');
        list.forEach(item => {
          uploadFileList.value[item.order].status = 'error';
          uploadFileList.value[item.order].errorText = data?.msg || common.upFailed;
        });
      }
    })
    .catch(error => {
      console.log(error);
      list.forEach(item => {
        uploadFileList.value[item.order].status = fileStatus.error;
        uploadFileList.value[item.order].errorText = item?.msg || common.upFailed;
      });
      message.error(JSON.stringify(error.message) || '出错了');
    });
  getFaqList();
};

// function getBase64(file: File) {
//   return new Promise((resolve, reject) => {
//     const reader = new FileReader();
//     reader.readAsDataURL(file);
//     reader.onload = () => resolve(reader.result);
//     reader.onerror = error => reject(error);
//   });
// }

// const handlePreview = async (file: UploadProps['fileList'][number]) => {
//   if (!file.url && !file.preview) {
//     file.preview = (await getBase64(file.originFileObj)) as string;
//   }
//   previewImage.value = file.url || file.preview;
//   previewVisible.value = true;
//   previewTitle.value = file.name || file.url.substring(file.url.lastIndexOf('/') + 1);
// };

const handleCancel = () => {
  previewVisible.value = false;
  previewTitle.value = '';
};
</script>
<style lang="scss" scoped>
.edit-qa-set-comp {
  width: 100%;
  height: 100%;
  font-family: PingFang SC;
  .tabs {
    // width: 156px;
    margin: 20px 0 16px 0;
    display: flex;
    .tab-item {
      font-size: 16px;
      color: #666666;
      cursor: pointer;
      margin-right: 28px;
    }
    .tab-item-active {
      font-weight: 500;
      color: #5a47e5;
    }
  }
  .upload-content {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    .upload-max {
      color: #999999;
    }
  }
  .upload-excel {
    // height: 248px;
    padding: 18px;
    margin-bottom: 24px;
    border-radius: 8px;
    background: #f9f9fc;
    .demo-download {
      margin-bottom: 16px;
      span {
        font-size: 14px;
        color: #999999;
      }
      a {
        font-size: 14px;
        font-weight: normal;
        color: #5a47e5;
        margin-right: 8px;
      }
    }
  }
  .box {
    height: 198px;
    border-radius: 8px;
    background: #fff;
    border: 1px dashed #dedcdc;
    .before-upload-box {
      position: relative;
      width: 100%;
      height: 100%;

      .hide {
        opacity: 0;
      }

      .input {
        position: absolute;
        width: 100%;
        height: 100%;
        z-index: 100;
      }
      .before-upload {
        width: 100%;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
      }
      .upload-text-box {
        display: flex;
        align-items: center;
        justify-content: center;

        svg {
          width: 16px;
          height: 16px;
          margin-right: 4px;
          cursor: pointer;
        }

        .upload-text {
          // font-weight: 500;
          font-size: 14px;
          color: $title1;
        }

        .blue {
          color: #5a47e5;
          cursor: pointer;
        }
      }

      .desc {
        color: $title3;
        text-align: center;
        margin-top: 8px;
        padding: 0 20px;
      }
    }

    .upload-box {
      &.upload-list {
        height: 188px;
      }

      .list {
        height: 188px;

        overflow: auto;

        li {
          display: flex;
          align-items: center;
          justify-content: space-around;
          height: 22px;
          margin-bottom: 20px;
          padding: 0 20px 0 16px;

          &:first-child {
            margin-top: 20px;
          }
          svg {
            width: 16px;
            height: 16px;
            margin-right: 4px;
          }

          .name {
            flex: 1;
            width: 0;
            margin-right: 20px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          .status-box {
            display: flex;
            width: auto;
            align-items: center;
            justify-content: start;
            margin-right: 5px;

            .loading {
              width: 16px;
              height: 16px;
              margin-right: 4px;
              animation: 2s linear infinite loading;
            }
            .status {
              width: 100px;
              font-size: 14px;
              line-height: 22px;
              height: 22px;
              color: $title1;
              overflow: hidden;
              text-overflow: ellipsis;
              white-space: nowrap;
            }
          }

          .delete {
            line-height: 22px;
            color: $title2;
            cursor: pointer;
          }
        }
      }

      .note {
        font-family: PingFang SC;
        font-size: 12px;
        font-weight: normal;
        margin-top: 12px;
        color: #999999;
        width: 330px;
      }
      .overly {
        display: none;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 200px;
        border-radius: 8px;
        font-size: 14px;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        cursor: pointer;
        svg {
          width: 20px;
          height: 20px;
          margin-bottom: 8px;
        }
        .overly-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }
        .hide {
          opacity: 0;
        }

        .input {
          position: absolute;
          width: 100%;
          height: 100%;
          z-index: 100;
        }
      }
    }

    .upload-2 {
      position: relative;
    }

    .upload-2:hover .overly {
      display: block;
    }

    :deep(.ant-input) {
      height: 40px;
    }
  }
  .footer {
    display: flex;
    justify-content: end;
    .cancel-btn {
      // width: 68px;
      height: 32px;
      padding: 0 20px;
      margin-right: 16px;
    }
    .login-form-btn {
      // width: 68px;
      height: 32px;
      padding: 0 20px;
      background: #5a47e5 !important;
    }
  }
  :deep(.ant-upload-list) {
    width: auto;
  }
}
</style>
