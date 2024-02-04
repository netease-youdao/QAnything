<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-01 10:59:31
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-02 11:22:55
 * @FilePath: /qanything-open-source/src/components/SiderCard.vue
 * @Description: 
-->
<template>
  <div
    v-for="(item, index) in props.list"
    :key="index"
    :class="{ active: selectList.includes(item.kb_id) }"
    class="card"
    :style="props.style"
    @click="selectKnowledgeBase(item)"
  >
    <a-popover overlay-class-name="card-hover" placement="right">
      <template #content>
        <div class="tools-box">
          <ul>
            <li @click="manage(item)">
              <SvgIcon class="edit" name="icon-manage"></SvgIcon>
              <span class="tool-name">{{ common.manage }}</span>
            </li>
            <li @click="editKnowledgeBase(item)">
              <SvgIcon class="edit" name="edit"></SvgIcon>
              <span class="tool-name">{{ common.rename }}</span>
            </li>
            <li @click="deleteKnowledgeBase(item)">
              <SvgIcon class="delete" name="delete"></SvgIcon>
              <span class="tool-name">{{ common.delete }}</span>
            </li>
          </ul>
        </div>
      </template>
      <div class="content">
        <div class="title">
          <div v-show="!item.edit" class="normal">
            <p class="title-text">{{ item.kb_name }}</p>
          </div>

          <div v-show="item.edit" class="editing">
            <p class="title-text">
              <a-input v-model:value="item.kb_name" type="text"></a-input>
            </p>
            <span class="icon-box">
              <SvgIcon class="edit" name="card-confirm" @click.stop="ok(item)"></SvgIcon>
              <SvgIcon class="delete" name="card-cancel" @click.stop="close(item)"></SvgIcon>
            </span>
          </div>
        </div>
        <div class="time">{{ item.createTime }}</div>
      </div>
    </a-popover>
  </div>
</template>
<script lang="ts" setup>
import { IKnowledgeItem } from '@/utils/types';
import { useKnowledgeBase } from '@/store/useKnowledgeBase';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import { pageStatus } from '@/utils/enum';
import { getLanguage } from '@/language/index';

const common = getLanguage().common;
// import { useDebounceFn } from '@vueuse/core';
const { setShowDeleteModal, setCurrentId, setCurrentKbName, setDefault } = useKnowledgeBase();
const { showDeleteModal, selectList } = storeToRefs(useKnowledgeBase());

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
        width: '232px',
      };
    },
  },
});

//点击编辑时候 记录标题 如果取消,展示oldValue
const oldValue = ref('');

//选择知识库
const selectKnowledgeBase = (item: IKnowledgeItem) => {
  if (item.edit) {
    //如果正在编辑知识库名字 此时点击知识库卡片不做操作
    return;
  }
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

//管理知识库
const manage = item => {
  setCurrentId(item.kb_id);
  setCurrentKbName(item.kb_name);
  setDefault(pageStatus.optionlist);
};

//删除知识库
const deleteKnowledgeBase = (item: IKnowledgeItem) => {
  setShowDeleteModal(!showDeleteModal.value);
  setCurrentId(item.kb_id);
  console.log(`删除${item.kb_id}`);
};

//修改知识库标题
const editKnowledgeBase = (item: IKnowledgeItem) => {
  console.log(`编辑${item.kb_id}`);
  setCurrentId(item.kb_id);
  oldValue.value = item.kb_name;
  item.edit = !item.edit;
};

//确定修改
const ok = async (item: IKnowledgeItem) => {
  try {
    await resultControl(
      await urlResquest.kbConfig({ kb_id: item.kb_id, new_kb_name: item.kb_name })
    );
    oldValue.value = '';
    item.edit = !item.edit;
    message.success(common.renameSucceeded);
  } catch (err) {
    message.error(err.msg || common.renameFailed);
  }
};
//取消修改
const close = (item: IKnowledgeItem) => {
  console.log('取消修改', item);
  item.kb_name = oldValue.value;
  oldValue.value = '';
  item.edit = !item.edit;
};
</script>
<style lang="scss" scoped>
.card {
  overflow: hidden;
  position: relative;
  height: 72px;
  margin: 0 auto;
  margin-bottom: 16px;
  border-radius: 8px;
  background: #333647;
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
    margin: 14px 1px 0px 12px;

    .normal {
      .title-text {
        width: 169px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        color: #ffffff;
      }
    }

    .editing {
      display: flex;
      align-items: center;
      .title-text {
        width: 160px;
        height: 28px;
        color: #ffffff;

        :deep(.ant-input) {
          padding: 2px 11px;
        }
      }

      .icon-box {
        color: #fff;

        .edit,
        .delete {
          width: 16px;
          height: 16px;
          cursor: pointer;
        }

        .edit {
          margin-left: 12px;
          margin-right: 8px;
        }
      }

      :deep(.ant-input) {
        color: #999999;
        background: #1e212f;
        border-color: #7261e9;
      }
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
  background: #7261e9;
}

.fade-enter-active {
  transition: all 0.3s ease-out;
}

.fade-leave-active {
  opacity: 0.5;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
<style lang="scss">
.content {
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.tools-box {
  background: #333647;
  ul {
    width: 100%;
    li {
      cursor: pointer;
      display: flex;
      width: 80px;
      margin-bottom: 3px;
      box-sizing: content-box;
      height: 28px;
      align-items: center;
      font-size: 14px;
      font-weight: normal;
      line-height: 22px;
      color: #c0c0c0;

      .tool-name {
        color: #c0c0c0;
      }

      svg {
        margin: 0 6px 0 8px;
        width: 16px;
        height: 16px;
      }

      &:hover {
        background: #1e212f;
        color: #fff;

        .tool-name {
          color: #fff;
        }
      }
    }
  }
}
.card-hover {
  .ant-popover-arrow {
    display: none;
  }
  .ant-popover-content {
    .ant-popover-inner {
      transform: translateY(10px);
      background: #333647;
      box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.2);
      padding: 4px;
    }
  }
}
</style>
