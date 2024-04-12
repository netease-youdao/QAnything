<template>
  <div class="bot-list">
    <div class="bot-item" @click="setNewBotsVisible(true)">
      <div class="new-bot-content">
        <img class="new-bot" src="@/assets/bots/new-bot-icon.png" alt="icon" />
        <div class="new-bot-text">{{ bots.createBot }}</div>
      </div>
    </div>
    <div class="bot-item" v-for="item in botList" :key="item.id" @click="botEdit(item)">
      <div class="top-info">
        <img class="avator" src="@/assets/bots/bot-avatar.png" alt="avator" />
        <span class="name">{{ item.name }}</span>
        <a-dropdown
          @click.stop
          :trigger="['click']"
          placement="bottomLeft"
          overlay-class-name="operate-select-menu"
        >
          <div class="more-icon">
            <img src="@/assets/bots/more-icon.png" alt="icon" />
          </div>
          <template #overlay>
            <a-menu>
              <a-menu-item>
                <div class="item" @click="deleteBot(item)">
                  <span class="delete-icon icon" />
                  <span class="text">{{ bots.delete }}</span>
                </div>
              </a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>
      </div>
      <div class="intro">{{ item.description }}</div>
      <div class="time">
        {{ bots.recentlyEdited }} {{ moment(item.updateTime).format('YYYY-MM-DD') }}
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import { useBotsChat } from '@/store/useBotsChat';
import routeController from '@/controller/router';
import urlResquest from '@/services/urlConfig';
import { resultControl } from '@/utils/utils';
import { message } from 'ant-design-vue';
import moment from 'moment';
import { getLanguage } from '@/language/index';

const { botList } = storeToRefs(useBots());
const { setNewBotsVisible, setTabIndex, setCurBot } = useBots();
const { setQaList } = useBotsChat();
const { changePage } = routeController();

const emits = defineEmits(['getBotList']);
const bots = getLanguage().bots;

const botEdit = item => {
  console.log('botEdit', item);
  setTabIndex(0);
  setCurBot(item);
  setQaList([]);
  changePage(`/bots/${item.bot_id}/edit`);
};

const deleteBot = async data => {
  try {
    await resultControl(await urlResquest.deleteBot({ bot_id: data.bot_id }));
    emits('getBotList');
    message.success(bots.deletedSucessfully);
  } catch (e) {
    message.error(e.msg || '删除失败，请重试');
  }
};
</script>
<style lang="scss" scoped>
.bot-list {
  width: 100%;
  height: 100%;
  background: #f3f6fd;
  font-family: PingFang SC;
  padding: 32px;
  display: grid;
  // grid-template-columns: repeat(4, 1fr);
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px; /* 设置元素之间的间距 */
  .bot-item {
    width: 100%;
    height: 160px;
    border-radius: 12px;
    background: #fff;
    padding: 20px 28px;
    margin-right: 24px;
    margin-bottom: 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    cursor: pointer;
  }
  .bot-item:hover {
    transform: translateY(-10px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
  }
  .new-bot-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    .new-bot {
      width: 46px;
      height: 46px;
      margin-bottom: 10px;
    }
    .new-bot-text {
      font-size: 18px;
      font-weight: 500;
      color: #7261e9;
    }
  }
  .top-info {
    width: 100%;
    height: 32px;
    display: flex;
    margin-bottom: 12px;
    align-items: center;
    .avator {
      width: 42px;
      height: 42px;
    }
    .name {
      flex-grow: 1;
      font-size: 18px;
      font-weight: 500;
      color: #222222;
      margin-left: 12px;
      display: -webkit-box;
      -webkit-line-clamp: 1;
      -webkit-box-orient: vertical;
      overflow: hidden; /* 隐藏超出容器的文本 */
      text-overflow: ellipsis; /* 超出部分以省略号表示 */
    }
    .more-icon {
      width: 25px;
      height: 25px;
      display: flex;
      justify-content: center;
      align-items: center;
      border-radius: 5px;
      img {
        width: 24px;
        height: 24px;
      }
    }
    .more-icon:hover {
      background: #f4f5f7;
    }
    .dropdown-item {
      font-size: 14px;
      color: #666666;
      img {
        width: 16px;
        height: 16px;
        margin-right: 8px;
      }
    }
  }
  .intro {
    width: 100%;
    height: 38px;
    line-height: 20px;
    font-size: 14px;
    color: #666666;
    margin-bottom: 12px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden; /* 隐藏超出容器的文本 */
    text-overflow: ellipsis; /* 超出部分以省略号表示 */
  }
  .time {
    width: 100%;
    font-size: 12px;
    color: #999999;
  }

  .operate-select-menu {
    .ant-dropdown-menu {
      border-radius: 8px;
      background: #ffffff;
      border: 1px solid #f4f5f7;
      box-shadow: 0px 6px 30px 0px rgba(19, 36, 64, 0.08);
      padding: 0px;
    }
  }
}
</style>
<style lang="scss">
.operate-select-menu {
  .ant-dropdown-menu-item {
    .item {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      font-size: 14px;
      color: #666666;
    }
    .icon {
      display: inline-block;
      width: 16px;
      height: 16px;
      margin-right: 4px;
    }

    .delete-icon {
      background: url(@/assets/bots/delete-icon.png) no-repeat center center;
      background-size: 100%;
    }
  }
}
</style>
