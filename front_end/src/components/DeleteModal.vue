<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 17:55:20
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-12-27 11:52:28
 * @FilePath: /qanything-open-source/src/components/DeleteModal.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <Teleport to="body">
    <div class="private">
      <a-modal
        v-model:open="showDeleteModal"
        :confirm-loading="confirmLoading"
        centered
        width="480px"
        wrap-class-name="private-modal"
        :ok-text="common.confirm"
        :cancel-text="common.cancel"
        @ok="handleOk"
      >
        <template #title>
          <div class="private-title">
            <SvgIcon class="info" name="info"></SvgIcon>{{ common.deleteDec }}
          </div>
        </template>
      </a-modal>
    </div>
  </Teleport>
</template>
<script lang="ts" setup>
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import urlResquest from '@/services/urlConfig';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;
// import { resultControl } from '@/utils/utils';
const { showDeleteModal, currentId, selectList } = storeToRefs(useKnowledgeBase());
const { setShowDeleteModal, getList } = useKnowledgeBase();
const confirmLoading = ref<boolean>(false);
const handleOk = async () => {
  confirmLoading.value = true;

  console.log('调用删除');
  try {
    const res = await urlResquest.deleteKB({ kb_ids: [currentId.value] });
    if (+res.code === 200) {
      const index = selectList.value.findIndex(item => {
        return item === currentId.value;
      });
      if (index != -1) {
        selectList.value.splice(index, 1);
      }
      setShowDeleteModal(false);
      confirmLoading.value = false;
      message.success('删除成功');
      getList();
    } else {
      confirmLoading.value = false;
      message.error('删除失败');
    }
  } catch (e) {
    console.log(e);
    confirmLoading.value = false;
    message.error(e.msg || '删除失败');
  }
};
</script>
<style lang="scss" scoped>
.info {
  width: 24px;
  height: 24px;
}

.private-title {
  display: flex;
  align-items: center;
  font-size: 16px;
  font-weight: normal;
  line-height: 24px;
  color: #2e2f33;
  .info {
    margin-right: 12px;
  }
}
</style>

<style lang="scss">
//删除弹框相关修改
.private-modal {
  background: rgba(0, 0, 0, 0.7);
  .ant-modal-content {
    padding: 32px 24px 18px 24px;
    font-size: 16px;
    font-weight: 500;
    color: $title1;
  }
  .ant-modal-title {
    margin-bottom: 8px;
  }
  .ant-modal-body {
    margin-left: 40px;
    font-size: 14px;
    color: $title1;
  }
  .ant-modal-footer {
    margin-top: 58px;
    .ant-btn {
      border-radius: 4px;
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      padding: 5px 20px;
      border-color: #dfe3eb;
      color: #222222;
    }

    .ant-btn-primary {
      background: #5a47e5 !important;
      color: #ffffff;
    }
  }

  .ant-modal-close {
    width: 16px;
    height: 16px;
  }
  .ant-modal-close-x {
    line-height: 16px;
  }
}
</style>
