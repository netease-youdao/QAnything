<template>
  <Teleport to="body">
    <a-modal
      v-model:open="selectKnowledgeVisible"
      :title="bots.selectKb"
      centered
      :destroyOnClose="true"
      width="600px"
      wrap-class-name="select-knowledge-modal"
      :footer="null"
    >
      <div class="select-kenoledge-comp">
        <div class="header">
          <a-input class="select-input" v-model:value="knowledge" :placeholder="bots.search">
            <template #prefix>
              <img class="search-icon" src="@/assets/bots/search.png" alt="search-icon" />
            </template>
          </a-input>
          <a-button class="btn" type="primary" @click="toCreatekb">{{ bots.createKb }}</a-button>
        </div>
        <div class="content">
          <div
            class="knowledge-item"
            v-for="item in knowledgeList.filter(item => regex.test(item.kb_nmame))"
            :key="item.kbId"
          >
            <img src="@/assets/bots/knowledge.png" alt="knowledge" />
            <div class="detail-info">
              <div class="kb-name">{{ item.kb_name }}</div>
              <!-- <div class="kb-time">{{ bots.creationTime }} {{ item.time }}</div> -->
            </div>
            <div
              :class="['button', `button-${item.state}`, `button-${common.type}-${item.state}`]"
              @click="handleKbBind(item)"
            ></div>
          </div>
          <div class="bottom-text">{{ bots.noMore }}</div>
        </div>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import { useHeader } from '@/store/useHeader';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import routeController from '@/controller/router';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';

const { changePage } = routeController();
const { selectKnowledgeVisible, knowledgeList, curBot } = storeToRefs(useBots());
const { setCurBot } = useBots();
const { setNavIndex } = useHeader();
const knowledge = ref('');
const bots = getLanguage().bots;
const common = getLanguage().common;
const regex = computed(() => new RegExp(knowledge.value, 'i'));

const getBotInfo = async botId => {
  try {
    const res: any = await resultControl(await urlResquest.queryBotInfo({ bot_id: botId }));
    setCurBot(res[0]);
  } catch (e) {
    message.error(e.msg || '获取Bot信息失败');
  }
};

const handleKbBind = async data => {
  const kbIds = curBot.value.kb_ids;
  kbIds.push(data.kb_id);
  console.log('kbIds', kbIds);
  try {
    await resultControl(
      await urlResquest.updateBot({
        bot_id: curBot.value.bot_id,
        kb_ids: kbIds,
      })
    );
    getBotInfo(curBot.value.bot_id);
    knowledgeList.value = knowledgeList.value.map(item => {
      if (item.kb_id === data.kb_id) {
        item.state = item.state === 0 ? 1 : 0;
      }
      return item;
    });
  } catch (e) {
    message.error(e.msg || '请求失败');
  }
};

const toCreatekb = () => {
  setNavIndex(0);
  changePage('/home');
  selectKnowledgeVisible.value = false;
};
</script>
<style lang="scss" scoped>
.select-kenoledge-comp {
  width: 100%;
  height: 100%;
  font-family: PingFang SC;
  .header {
    width: 100%;
    height: 32px;
    margin-bottom: 20px;
    display: flex;
    .select-input {
      width: 280px;
      margin-right: 12px;
    }
    .search-icon {
      width: 16px;
      height: 16px;
    }
    .btn {
      // width: 94px;
      height: 32px;
      padding: 0 12px !important;
      text-align: center;
      background: #5a47e5 !important;
    }
  }
  .content {
    width: 100%;
    height: 280px;
    overflow: auto;
    border-top: 1px solid #ededed;
    .knowledge-item {
      width: calc(100% - 48px);
      height: 80px;
      padding: 0 28px 0 20px;
      display: flex;
      align-items: center;
      img {
        width: 32px;
        height: 32px;
        margin-right: 12px;
      }
      .detail-info {
        flex-grow: 1;
        .kn-name {
          font-size: 14px;
          color: #222222;
        }
        .kb-time {
          font-size: 12px;
          color: #999999;
        }
      }
      .button {
        width: 68px;
        height: 32px;
        border-radius: 4px;
        background: #fff;
        box-sizing: border-box;
        border: 1px solid #dfe3eb;
        font-size: 14px;
        text-align: center;
        line-height: 32px;
        color: #5a47e5;
        cursor: pointer;
      }
      .button-0::after {
        content: '添加';
      }
      .button-en-0::after {
        content: 'Add';
      }
      .button-1 {
        opacity: 0.4;
      }
      .button-1::after {
        content: '已添加';
      }
      .button-en-1::after {
        content: 'Added';
      }
      .button-1:hover {
        color: #ff3838;
        opacity: 1;
      }
      .button-1:hover::after {
        content: '移除';
      }
      .button-en-1:hover::after {
        content: 'Remove';
      }
    }
    .knowledge-item:hover {
      background: #e9edf7;
    }
    .bottom-text {
      width: 100%;
      text-align: center;
      font-size: 14px;
      color: #9e9e9e;
      margin-top: 16px;
    }
  }
}
</style>
