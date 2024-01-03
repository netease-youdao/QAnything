<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 14:57:33
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-29 16:12:11
 * @FilePath: /qanything-open-source/src/components/Sider.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div class="sider">
    <div class="add-btn">
      <!-- <AddInput @add="addKb" /> -->
      <AddInput />
    </div>
    <div class="content">
      <SiderCard :list="knowledgeBaseList"></SiderCard>
    </div>
    <!-- <div class="bottom-btn-box">
      <a-button class="manage" @click="goManage">
        <template #icon>
          <img class="folder" src="../assets/home/icon-folder.png" alt="图标" />
        </template>
        知识库管理</a-button
      >
    </div> -->
    <DeleteModal />
    <FileUploadDialog />
    <UrlUploadDialog />
  </div>
</template>
<script lang="ts" setup>
// import dayjs from 'dayjs';
import AddInput from '@/components/AddInput.vue';
import SiderCard from '@/components/SiderCard.vue';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import FileUploadDialog from '@/components/FileUploadDialog.vue';
import UrlUploadDialog from '@/components/UrlUploadDialog.vue';
import DeleteModal from '@/components/DeleteModal.vue';
// import { useKnowledgeModal } from '@/store/useKnowledgeModal';
// import { message } from 'ant-design-vue';
// import urlResquest from '@/services/urlConfig';
// import { pageStatus } from '@/utils/enum';

// const { setModalVisible } = useKnowledgeModal();
// const { modalVisible } = storeToRefs(useKnowledgeModal());

// const router = useRouter();
// const { getList, setCurrentId, setCurrentKbName, setDefault } = useKnowledgeBase();
// const { knowledgeBaseList, selectList } = storeToRefs(useKnowledgeBase());
const { knowledgeBaseList } = storeToRefs(useKnowledgeBase());

//创建知识库
// const addKb = async kb_name => {
//   console.log(kb_name);
//   if (!kb_name.length) {
//     message.error('请输入知识库名称');
//     return;
//   }

//   try {
//     const res: any = await urlResquest.createKb({ kb_name: kb_name });
//     if (+res.code === 200) {
//       console.log(res);
//       setCurrentId(res?.data?.kb_id);
//       setCurrentKbName(res?.data?.kb_name);
//       selectList.value.push(res.data.kb_id);
//       await getList();
//       setModalVisible(!modalVisible.value);
//       setDefault(pageStatus.optionlist);
//     }
//   } catch (e) {
//     message.error(e.msg || '请求失败');
//   }
// };
</script>

<style lang="scss" scoped>
.sider {
  display: flex;
  flex-direction: column;
  width: 280px;
  height: calc(100vh - 64px);
  background-color: $baseColor;

  .add-btn {
    margin: 28px 24px 20px 24px;
    width: calc(100% - 48px);

    :deep(.ant-input-affix-wrapper) {
      padding: 4px;
      border: 1px solid #373b4d;
      background: linear-gradient(0deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), #26293b;
    }

    :deep(.ant-input) {
      color: #ffffff;
      padding-left: 4px;
      background: linear-gradient(0deg, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), #26293b;
      &::placeholder {
        color: #999999;
      }
    }
  }

  .bottom-btn-box {
    position: fixed;
    width: 280px;
    bottom: 29px;
    .manage {
      width: calc(100% - 40px);
      margin: 0 20px;
      height: 40px;
    }

    .folder {
      width: 16px;
      height: 16px;
      margin-right: 8px;
    }

    :deep(.ant-btn) {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #4d71ff !important;
      background: rgba(255, 255, 255, 0.7) !important;
      border: 1px solid #ffffff !important;
    }
  }
}

.content {
  flex: 1;
  margin-bottom: 20px;
  margin-top: 20px;
  overflow-y: scroll;
}
</style>
