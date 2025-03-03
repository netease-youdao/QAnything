<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 10:59:31
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2023-11-29 12:18:02
 * @FilePath: /ai-demo/src/components/ManageCard.vue
 * @Description: 
-->
<template>
  <template v-if="props.list && props.list.length > 0">
    <div
      v-for="(item, index) in props.list"
      :key="index"
      :class="{ active: selectList.includes(item.kb_id) }"
      class="card"
      :style="props.style"
      @click="selectKnowledgeBase(item)"
    >
      <div class="title">
        <div class="normal">
          <p class="title-text">{{ item.kb_name }}</p>
          <span v-show="selectList.includes(item.kb_id)" class="icon-box">
            <SvgIcon class="edit" name="edit" @click.stop="editKnowledge(item)"></SvgIcon>
            <SvgIcon class="delete" name="delete" @click.stop="deleteKnowledgeBase(item)"></SvgIcon>
          </span>
        </div>
      </div>
      <!-- <div class="time">{{ item.createTime }}</div> -->
    </div>
  </template>

  <!-- <div v-else class="no-data">
    <a-empty :image="simpleImage" />
  </div> -->
</template>
<script lang="ts" setup>
// import { Empty } from 'ant-design-vue';
// const simpleImage = Empty.PRESENTED_IMAGE_SIMPLE;
import { IKnowledgeItem } from '@/utils/types';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import { useKnowledgeModal } from '@/store/useKnowledgeModal';
import urlResquest from '@/services/urlConfig';
import { getStatus } from '@/utils/utils';
import { message } from 'ant-design-vue';
const { setModalVisible, setModalTitle, setFileList } = useKnowledgeModal();
const { modalVisible } = storeToRefs(useKnowledgeModal());

// import { useDebounceFn } from '@vueuse/core';
const { setCurrentId, setShowDeleteModal, setCurrentKbName } = useKnowledgeBase();
const { currentId, selectList, showDeleteModal } = storeToRefs(useKnowledgeBase());

const props = defineProps({
  list: {
    type: Array<IKnowledgeItem>,
    require: true,
    default: () => [],
  },
  style: {
    type: Object,
    default: () => {
      return {
        width: 'calc(100% - 40px)',
      };
    },
  },
});

//选择知识库
const selectKnowledgeBase = (item: IKnowledgeItem) => {
  const id = item.kb_id;
  console.log('点击知识库');
  if (selectList.value.includes(id)) {
    const index = selectList.value.findIndex(Iitem => {
      return Iitem === id;
    });

    if (index != -1) {
      selectList.value.splice(index, 1);
    }
  } else {
    selectList.value.push(id);
  }
};

//编辑知识库
const editKnowledge = async (item: IKnowledgeItem) => {
  console.log('编辑知识库');
  setCurrentId(item.kb_id);
  //获取当前id的知识库 详细信息
  await getDetails();
  setCurrentKbName(item.kb_name);
  setModalTitle('编辑知识库');
  setModalVisible(!modalVisible.value);
};

//删除知识库
const deleteKnowledgeBase = (item: IKnowledgeItem) => {
  setCurrentId(item.kb_id);
  setShowDeleteModal(!showDeleteModal.value);
  console.log(showDeleteModal.value);
  console.log(`删除${item.kb_id}`);
};

const getDetails = async () => {
  try {
    const res: any = await urlResquest.fileList({ kb_id: currentId.value });
    console.log('filelist-res', res);
    if (+res.code === 200) {
      res.data.details.forEach((item: any) => {
        item.errorText = getStatus(item);
      });

      setFileList(res.data.details);
    } else {
      message.error('获取知识库详情失败');
    }
  } catch (error) {
    message.error(error.msg || '获取知识库详情失败');
  }
};
</script>
<style lang="scss" scoped>
.card {
  overflow: hidden;
  position: relative;
  height: 72px;
  margin: 0 20px;
  margin-bottom: 16px;
  border-radius: 8px;
  background: #fff;
  cursor: pointer;
  border: 1px solid transparent;
  user-select: none;

  .title {
    display: flex;
    align-items: center;
    color: $title1;
    font-size: 14px;
    height: 22px;
    line-height: 22px;
    margin: 14px 1px 0px 16px;

    .normal {
      .title-text {
        width: 169px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
      }
    }

    .editing {
      margin-right: 8px;

      .title-text {
        width: 168px;
        height: 28px;

        :deep(.ant-input) {
          padding: 2px 11px;
        }
      }

      .icon-box {
        right: 8px;
      }

      :deep(.ant-input) {
        background: #f3f3f3;
        border-color: #f3f3f3;
      }
    }

    .edit,
    .delete {
      width: 16px;
      height: 16px;
      cursor: pointer;
    }

    .edit {
      margin-right: 8px;
    }

    .icon-box {
      position: absolute;
      right: 14px;
      top: 17px;
    }
  }

  .time {
    margin-top: 2px;
    margin-left: 16px;
    color: $title3;
    font-size: 12px;
    height: 20px;
    line-height: 20px;
  }
}
.active {
  border: 2px solid #4d71ff;
}
.no-data {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
</style>
