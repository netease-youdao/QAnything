<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-07 19:32:26
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-08-05 15:34:11
 * @FilePath: front_end/src/components/FileUploadDialog.vue
 * @Description:
-->
<template>
  <Teleport to="body">
    <a-modal
      v-model:open="modalVisible"
      :title="modalTitle"
      centered
      width="480px"
      wrap-class-name="upload-file-modal"
      :destroy-on-close="true"
    >
      <div class="file">
        <div class="box">
          <div class="before-upload-box" :class="showUploadList ? 'uploading' : ''">
            <input
              class="hide input"
              type="file"
              :accept="acceptList.join(',')"
              multiple
              @change="fileChange"
              @click="e => ((e.target as HTMLInputElement).value = '')"
            />
            <div class="before-upload">
              <div class="upload-text-box">
                <SvgIcon name="upload" />
                <p>
                  <span class="upload-text">
                    {{ common.dragUrl }}
                    <span class="blue">{{ common.click }}</span>
                  </span>
                </p>
              </div>
              <p v-if="!showUploadList" class="desc">
                {{ common.updesc1 }}
              </p>
            </div>
          </div>
          <div
            v-show="showUploadList"
            class="upload-box"
            :class="showUploadList ? 'upload-list' : ''"
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
                      <span class="status">{{
                        item.status == 'loading' ? item.text : item.errorText
                      }}</span>
                    </div>
                  </li>
                </ul>
              </template>
            </UploadList>
            <div class="note">{{ common.errorTip }}</div>
          </div>
        </div>
      </div>
      <template #footer>
        <a-button
          v-if="props.dialogType === 0"
          key="submit"
          type="primary"
          class="upload-btn"
          :disabled="!canSubmit"
          @click="handleOk"
        >
          {{ common.confirm }}
        </a-button>
        <a-button
          v-if="props.dialogType === 1"
          key="submit"
          type="primary"
          class="upload-btn"
          @click="handleCancel"
        >
          {{ common.cancel }}
        </a-button>
      </template>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { apiBase } from '@/services';
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { useOptiionList } from '@/store/useOptiionList';
import SvgIcon from './SvgIcon.vue';
import UploadList from '@/components/UploadList.vue';
import { pageStatus } from '@/utils/enum';
import { IFileListItem } from '@/utils/types';
import { message } from 'ant-design-vue';
import { userId } from '@/services/urlConfig';
import { getLanguage } from '@/language/index';
import { useUploadFiles } from '@/store/useUploadFiles';
// import { useLanguage } from '@/store/useLanguage';

// const { language } = storeToRefs(useLanguage());
const common = getLanguage().common;
const { setKnowledgeName, setModalVisible } = useKnowledgeModal();
const { setDefault } = useKnowledgeBase();
const { getDetails } = useOptiionList();
const { modalVisible, modalTitle } = storeToRefs(useKnowledgeModal());
const { currentId, currentKbName } = storeToRefs(useKnowledgeBase());
const { uploadFileList } = storeToRefs(useUploadFiles()); // 上传的文件列表
const { initUploadFileList } = useUploadFiles();

const props = defineProps({
  // 0为知识库上传，1为快速开始上传
  dialogType: {
    type: Number,
    require: false,
    default: 0,
  },
});

const timer = ref();

// const uploadFileList = ref([]); // 本次上传文件列表

//控制确认按钮 是否能提交
const canSubmit = computed(() => {
  console.log(currentId.value, uploadFileList.value);
  return (
    currentId.value.length > 0 &&
    uploadFileList.value.length > 0 &&
    uploadFileList.value.every(item => item.status != 'loading')
  );
});

watch(
  () => modalVisible.value,
  () => {
    setKnowledgeName(currentKbName.value);
    showUploadList.value = !!uploadFileList.value.length;
    if (!modalVisible.value && props.dialogType === 0) {
      initUploadFileList();
    }
  }
);

//是否显示上传文件列表 默认不显示
const showUploadList = ref(false);

//允许上传的文件格式
const acceptList = [
  '.md',
  '.txt',
  '.pdf',
  '.jpg',
  '.png',
  '.jpeg',
  '.docx',
  '.xlsx',
  '.pptx',
  '.eml',
  '.csv',
  // '.mp3',
  // '.wav',
];

//上传前校验
const beforeFileUpload = async (file, index) => {
  return new Promise((resolve, reject) => {
    if (file.name && acceptList.includes('.' + file.name.split('.').pop().toLowerCase())) {
      uploadFileList.value.push({
        file_name: file.name,
        file: file,
        status: 'loading',
        text: common.uploading,
        file_id: '',
        order: uploadFileList.value.length,
      });
      // initCurUploadFileNum();
      resolve(index);
    } else {
      reject(file.name);
    }
  });
};

//input上传
const fileChange = e => {
  const files = e.target.files;
  console.log('fileChange', files);
  Array.from(files).forEach(async (file: any, index) => {
    try {
      await beforeFileUpload(file, index);
    } catch (e) {
      message.error(`${e}的文件格式不符`);
      // uploadFileList.value.push({
      //   file_name: file.name,
      //   file: file,
      //   status: 'error',
      //   text: '文件格式不符',
      //   file_id: '',
      // });
    }
  });
  setTimeout(() => {
    uploadFileList.value.length && uplolad();
  });
};

// const uplolad = () => {
//   showUploadList.value = true;
//   uploadFileList.value.forEach(async (file: IFileListItem, index) => {
//     if (file.status == 'loading') {
//       try {
//         // 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传
//         const param = { files: file.file, kb_id: newId.value, mode: 'strong' };
//         console.log(param);
//         const res = await urlResquest.uploadFile(param, {
//           headers: {
//             'Content-Type': 'multipart/form-data',
//           },
//         });
//         if (+res.code === 200 && res.data[0].status !== 'red' && res.data[0].status !== 'yellow') {
//           uploadFileList.value[index].status = 'success';
//           uploadFileList.value[index].text = '上传成功';
//           uploadFileList.value[index].file_id = res.data[0].file_id;
//         } else {
//           uploadFileList.value[index].status = 'error';
//           uploadFileList.value[index].text = '上传失败';
//         }
//       } catch (e) {
//         uploadFileList.value[index].status = 'error';
//         uploadFileList.value[index].text = '上传失败';
//       }
//     }
//   });
// };

const uplolad = async () => {
  if (props.dialogType === 0) {
    showUploadList.value = true;
  }
  // else if (props.dialogType === 1) {
  //   handleCancel();
  // }
  const list = [];
  uploadFileList.value.forEach((file: IFileListItem) => {
    if (file.status == 'loading') {
      list.push(file);
    }
  });
  const formData = new FormData();
  for (let i = 0; i < list.length; i++) {
    formData.append('files', list[i]?.file);
  }
  formData.append('kb_id', currentId.value);
  formData.append('user_id', userId);
  // 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传
  formData.append('mode', 'strong');

  fetch(apiBase + '/local_doc_qa/upload_files', {
    method: 'POST',
    body: formData,
  })
    .then(response => {
      if (response.ok) {
        return response.json(); // 将响应解析为 JSON
      } else {
        throw new Error('上传失败');
      }
    })
    .then(data => {
      console.log('uploadList', uploadFileList.value, data);
      // 在此处对接口返回的数据进行处理
      if (data.code === 200) {
        if (data.data.length === 0) {
          message.warn(data.msg || '出错了');
          handleCancel();
          list.forEach(item => {
            uploadFileList.value[item.order].status = 'error';
            uploadFileList.value[item.order].errorText = data?.msg || common.upFailed;
          });
          return;
        }
        list.forEach((item, index) => {
          let status = data.data[index].status;
          if (status == 'green' || status == 'gray') {
            status = 'success';
          } else {
            status = 'error';
          }
          uploadFileList.value[item.order].status = status;
          console.log('data', data, item);
          uploadFileList.value[item.order].file_id = data.data[index].file_id;
          uploadFileList.value[item.order].errorText = common.upSucceeded;
        });
      } else {
        message.error(data.msg || '出错了');
        list.forEach(item => {
          uploadFileList.value[item.order].status = 'error';
          uploadFileList.value[item.order].errorText = data?.msg || common.upFailed;
        });
      }
    })
    .finally(() => {
      getDetails();
    });
};

const handleOk = async () => {
  setModalVisible(false);
  setDefault(pageStatus.optionlist);
  await getDetails();
};

const handleCancel = () => {
  setModalVisible(false);
};

onBeforeUnmount(() => {
  if (timer.value) {
    clearTimeout(timer.value);
  }
});
</script>
<style lang="scss" scoped>
.file {
  margin-top: 16px;
  display: flex;

  .box {
    flex: 1;
    height: 248px;
    border-radius: 6px;
    background: #f9f9fc;
    box-sizing: border-box;
    border: 1px dashed #ededed;
  }
}

.line-url {
  margin-top: 16px;
  height: 100px;
  display: flex;
  overflow: auto;

  .mt9 {
    margin-top: 9px;
  }

  :deep(.ant-input) {
    height: 30px;
  }

  :deep(.ant-form-item) {
    margin-bottom: 12px;
  }
}

.label {
  display: block;
  width: 82px;
  min-width: 82px;
  text-align: right;
  margin-right: 16px;
  color: $title1;

  .red {
    color: red;
  }
}

.before-upload-box {
  position: relative;
  width: 100%;
  height: 100%;

  &.uploading {
    height: 62px;
    border-bottom: 1px solid #ededed;
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
      font-weight: 500;
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
          width: 60px;
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
}

:deep(.ant-input) {
  height: 40px;
}

.upload-btn {
  background: #5147e5 !important;
}
</style>
<style lang="scss">
@keyframes loading {
  0% {
    transform: rotate(0deg);
  }

  50% {
    transform: rotate(180deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style>
