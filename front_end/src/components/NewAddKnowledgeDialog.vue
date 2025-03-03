<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-07 19:32:26
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-11 16:06:52
 * @FilePath: /ai-demo/src/components/NewAddKnowledgeDialog.vue
 * @Description:
-->
<template>
  <Teleport to="body">
    <a-modal
      v-model:open="modalVisible"
      :confirm-loading="confirmLoading"
      :title="modalTitle"
      :ok-button-props="{ disabled: !canSubmit }"
      centered
      width="740px"
      wrap-class-name="add-knowledge-modal"
      @ok="handleOk"
      @cancel="handleCancel"
    >
      <div class="kb-name">
        <span class="label"><span class="red">*</span> 知识库名称</span>
        <a-input v-model:value.trim="knowledgeName" :disabled="true" placeholder="请输入知识库名称">
          <template #suffix>
            <SvgIcon
              v-show="!newId.length && modalTitle !== '编辑知识库'"
              name="card-confirm"
              @click="addKnowledge"
            ></SvgIcon>
          </template>
        </a-input>
      </div>
      <div v-show="showUpload" class="file">
        <span class="label"><span class="red">*</span> 上传文件</span>
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
              <p class="desc">
                {{ common.updesc2 }}
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
                  <li v-for="(item, index) in fileList" :key="index">
                    <span class="name">{{ item.file_name }}</span>
                    <div class="status-box">
                      <SvgIcon
                        v-if="item.status != 'loading'"
                        :name="
                          item.status == 'gray' || item.status == 'green' ? 'success' : 'error'
                        "
                      />
                      <img
                        v-else
                        class="loading"
                        src="../assets/home/icon-loading.png"
                        alt="loading"
                      />
                      <span class="status">{{ getStatus(item) }}</span>
                    </div>
                    <span class="delete" @click="deleteFile(item, index)">
                      <svg-icon name="delete" />
                    </span>
                  </li>
                </ul>
              </template>
            </UploadList>
          </div>
        </div>
      </div>
      <div v-show="showUpload" class="line-url">
        <span class="label mt9">{{ common.addUrl }}</span>
        <UPloadInput :kb-id="newId"></UPloadInput>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import SvgIcon from './SvgIcon.vue';
import UploadList from '@/components/UploadList.vue';
import UPloadInput from '@/components/UploadInput.vue';
// import { fileStatus } from '@/utils/enum';
import { getStatus, resultControl } from '@/utils/utils';
import { IFileListItem } from '@/utils/types';
import urlResquest from '@/services/urlConfig';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';
import { useLanguage } from '@/store/useLanguage';

const { language } = storeToRefs(useLanguage());
const common = getLanguage().common;

const { $reset, setKnowledgeName } = useKnowledgeModal();
const { modalVisible, modalTitle, fileList, urlList, knowledgeName } = storeToRefs(
  useKnowledgeModal()
);
const { currentId, currentKbName } = storeToRefs(useKnowledgeBase());
const { getList } = useKnowledgeBase();

const confirmLoading = ref<boolean>(false);

const timer = ref();

//控制确认按钮 是否能提交
const canSubmit = computed(() => {
  return (
    knowledgeName.value.length > 0 &&
    newId.value.length > 0 &&
    fileList.value.length > 0 &&
    fileList.value.every(item => item.status != 'loading') &&
    urlList.value.every(
      item =>
        (item.status === 'default' && !item.text.length) ||
        item.status === 'success' ||
        item.status === 'default'
    )
  );
});
//控制知识库名称能否修改
// const canInput = computed(() => {
//   return newId.value.length > 0 && modalTitle.value === '新建知识库';
// });

const showTips = () => {
  return fileList.value.find(item => item.status === 'gray');
};

//新建完成后的知识库id
const newId = ref('');

watch(
  () => modalVisible.value,
  () => {
    if (modalTitle.value === '编辑知识库') {
      newId.value = currentId.value;
      setKnowledgeName(currentKbName.value);
    }
    if (fileList.value.length) {
      showUploadList.value = true;
    }
  }
);

//新建时候 没id不显示上传模块(原因：底层需要先创建知识库 才能长传内容)
const showUpload = computed(() => {
  return (newId.value && modalTitle.value === '新建知识库') || modalTitle.value !== '新建知识库';
});

//是否显示上传文件列表 默认不显示
const showUploadList = ref(false);

//允许上传的文件格式
const acceptList = [
  '.doc',
  '.docx',
  '.ppt',
  '.pptx',
  '.xls',
  '.xlsx',
  '.pdf',
  '.md',
  '.jpg',
  '.jpeg',
  '.png',
  '.bmp',
  '.txt',
  '.eml',
];

//上传前校验
const beforeFileUpload = async (file, index) => {
  return new Promise((resolve, reject) => {
    if (file.name && acceptList.includes('.' + file.name.split('.').pop().toLowerCase())) {
      fileList.value.push({
        file_name: file.name,
        file: file,
        status: 'loading',
        percent: 0,
        errorText: '',
        file_id: '',
      });
      resolve(index);
    } else {
      reject(file.name);
    }
  });
};

//input上传
const fileChange = e => {
  const files = e.target.files;
  Array.from(files).forEach(async (file: any, index) => {
    try {
      await beforeFileUpload(file, index);
    } catch (e) {
      message.error(`${e}的文件格式不符`);
    }
  });
  setTimeout(() => {
    const flag = fileList.value.find(item => {
      return item.status === 'loading';
    });
    flag && uplolad();
  });
};

const uplolad = () => {
  showUploadList.value = true;
  fileList.value.forEach(async (file: IFileListItem, index) => {
    if (file.status == 'loading') {
      try {
        // 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传
        const param = { files: file.file, kb_id: newId.value, mode: 'strong' };
        console.log(param);
        const res = await urlResquest.uploadFile(param, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        if (+res.code === 200 && res.data[0].status !== 'red' && res.data[0].status !== 'yellow') {
          fileList.value[index].status = res.data[0].status;
          fileList.value[index].file_id = res.data[0].file_id;
        } else {
          fileList.value[index].status = res.data[0].status;
          fileList.value[index].errorText = '上传失败';
        }
      } catch (e) {
        fileList.value[index].status = 'red';
        fileList.value[index].errorText = '上传失败';
      }
    }
  });
};

const deleteFile = async (item: IFileListItem, index: number) => {
  console.log(item.file_id);
  if (item.status == 'loading') {
    //上传过程中不能删除
    return message.info('上传过程中不能删除');
  }

  if (item.file_id) {
    try {
      const res = await urlResquest.deleteFile({ file_ids: [item.file_id], kb_id: newId.value });
      if (+res.code === 200) {
        fileList.value.splice(index, 1);
        message.success('删除成功');
      } else {
        message.error('删除失败');
      }
    } catch (e) {
      message.error(e.msg || '删除失败');
    }
  } else {
    //上传失败的 没有fileId  直接从filelist删除
    fileList.value.splice(index, 1);
    message.success('删除成功');
  }
};

const reset = () => {
  if (newId.value) {
    newId.value = '';
    getList();
  }
  console.log(showTips());
  if (showTips()) {
    message.warning('请重新进入“编辑知识库”查看解析结果');
  }
  $reset();
  showUploadList.value = false;
};

const handleOk = async () => {
  confirmLoading.value = true;
  console.log('确认知识库信息');
  if (modalTitle.value === '新建知识库') {
    timer.value = setTimeout(() => {
      confirmLoading.value = false;
      reset();
    }, 1000);
  } else {
    //编辑知识库
    if (knowledgeName.value !== currentKbName.value) {
      //修改修改知识库名字
      try {
        await resultControl(
          await urlResquest.kbConfig({ kbId: newId.value, kbName: knowledgeName.value })
        );
        confirmLoading.value = false;
        reset();
      } catch (e) {
        message.error(e.msg || '出错了');
      }
    } else {
      //没有修改知识库名字
      timer.value = setTimeout(() => {
        confirmLoading.value = false;
        reset();
      }, 1000);
    }
  }
};

const handleCancel = () => {
  console.log('取消知识库信息');
  reset();
};

//新建知识库
const addKnowledge = async () => {
  if (!knowledgeName.value.length) {
    message.error('请输入知识库名称');
    return;
  }
  //获取到知识库id后  赋值给newId
  try {
    const res: any = await resultControl(
      await urlResquest.createKb({ kbName: knowledgeName.value })
    );
    if (res && res.kbId) {
      console.log(res);
      newId.value = res.kbId;
    }
  } catch (e) {
    message.error(e.msg || common.error);
  }
};

onBeforeUnmount(() => {
  if (timer.value) {
    clearTimeout(timer.value);
  }
});
</script>
<style lang="scss" scoped>
.kb-name {
  display: flex;
  align-items: center;

  svg {
    width: 16px;
    height: 16px;
    cursor: pointer;
    margin-right: 16px;
  }

  :deep(.ant-input-affix-wrapper) {
    height: 40px;
  }

  :deep(.ant-input) {
    width: 586px;
    height: auto;
  }
}

.file {
  margin-top: 16px;
  display: flex;
  .box {
    flex: 1;
    height: 300px;
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
    height: 111px;
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
      line-height: 22px;
      height: 22px;
      color: $title1;
    }

    .blue {
      color: $baseColor;
      cursor: pointer;
    }
  }

  .desc {
    color: $title3;
    line-height: 22px;
    height: 22px;
    text-align: center;
    margin-top: 8px;
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
        margin-right: 60px;

        .loading {
          width: 16px;
          height: 16px;
          margin-right: 4px;
          animation: 2s linear infinite loading;
        }
        .status {
          width: 160px;
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
}

:deep(.ant-input) {
  height: 40px;
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
