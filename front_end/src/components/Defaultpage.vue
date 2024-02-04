<template>
  <div class="default">
    <div class="box">
      <p class="title">
        <span>{{ home.homeTitle1 }}</span
        ><span>&nbsp;</span><span class="color">{{ home.homeTitle2 }}</span>
      </p>
      <p class="desc">{{ home.defaultDec }}</p>

      <UploadDom :accept-list="acceptList" @update="update" />
    </div>
  </div>
</template>
<script lang="ts" setup>
import UploadDom from '@/components/UploadDom.vue';
import { message } from 'ant-design-vue';
// import { IFileListItem } from '@/utils/types';
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import urlResquest from '@/services/urlConfig';
import { getLanguage } from '@/language/index';

const home = getLanguage().home;
const { setFileList } = useKnowledgeModal();
const { setModalVisible } = useKnowledgeModal();
const { setCurrentId, getList, setCurrentKbName } = useKnowledgeBase();

// const emits = defineEmits(['change']);

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

const newId = ref('');

const update = async () => {
  console.log('updata');
  try {
    const res: any = await urlResquest.createKb({ kb_name: home.defaultName });
    if (+res.code === 200) {
      newId.value = res.data.kb_id;
      setCurrentId(res.data.kb_id);
      setCurrentKbName(res.kb_name);
      setModalVisible(true);
      getList();
    }
  } catch (e) {
    setFileList([]);
    message.error(e.msg);
  }
};

// const uplolad = async () => {
//   console.log('开始上传');
//   fileList.value.forEach(async (file: IFileListItem, index) => {
//     if (file.status == 'loading') {
//       try {
//         // 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传
//         const param = { files: file.file, kb_id: newId.value, mode: 'strong' };
//         console.log('upload-param', param);
//         const res = await urlResquest.uploadFile(param, {
//           headers: {
//             'Content-Type': 'multipart/form-data',
//           },
//         });
//         if (+res.code === 200 && res.data[0].status !== 'red' && res.data[0].status !== 'yellow') {
//           fileList.value[index].status = res.data[0].status;
//           fileList.value[index].file_id = res.data[0].file_id;
//         } else {
//           fileList.value[index].status = res.data[0].status;
//           fileList.value[index].errorText = '上传失败';
//         }
//       } catch (e) {
//         fileList.value[index].status = 'red';
//         fileList.value[index].errorText = '上传失败';
//       }
//     }
//   });
// };
</script>
<style lang="scss" scoped>
.default {
  width: 100%;
  height: 100%;
}

.box {
  position: relative;
  width: 800px;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  box-sizing: border-box;

  .title {
    text-align: center;
    line-height: 32px;
    height: 32px;
    span {
      display: inline-block;
      font-family: PingFang SC;
      font-size: 32px;
      font-weight: 600;
    }

    .color {
      color: #5a47e5;
    }
  }

  .desc {
    font-family: PingFang SC;
    font-size: 16px;
    font-weight: normal;
    height: 24px;
    line-height: 24px;
    letter-spacing: 0em;
    color: $title2;
    text-align: center;
    margin-top: 24px;
    margin-bottom: 40px;
  }
}
</style>
