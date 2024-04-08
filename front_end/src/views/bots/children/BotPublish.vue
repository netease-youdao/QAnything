<template>
  <div class="bot-publish">
    <div class="publish-item">
      <div class="content">
        <img src="@/assets/bots/web.png" alt="web" />
        <div class="intro">
          <p class="title">{{ bots.web }}</p>
          <p class="text">{{ bots.webDesc }}</p>
        </div>
      </div>
      <div class="bottom">
        <div class="bottom-item" @click="copyUrl">
          <img src="@/assets/bots/copy.png" alt="icon" />
          {{ bots.copyLink }}
        </div>
        <div class="bottom-item" @click="previewExperience">
          <img src="@/assets/bots/preview.png" alt="icon" />
          {{ bots.previewExperience }}
        </div>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import routeController from '@/controller/router';
import { useBots } from '@/store/useBots';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';

const { getCurrentRoute } = routeController();
const { webUrl } = storeToRefs(useBots());
const { setWebUrl, setCopyUrlVisible } = useBots();
const botId = ref(null);

const bots = getLanguage().bots;

message.config({
  top: `100px`,
  duration: 2,
  maxCount: 3,
});

init();
function init() {
  const route = getCurrentRoute();
  botId.value = route.value.params.botId;
  const { origin, pathname } = window.location;
  setWebUrl(`${origin + pathname}#/bots/${botId.value}/share`);
}

const copyUrl = () => {
  setCopyUrlVisible(true);
  // navigator.clipboard.writeText(webUrl.value).then(() => {
  //   message.success('复制成功');
  // });
};

const previewExperience = () => {
  window.open(webUrl.value, '_blank');
};
</script>
<style lang="scss" scoped>
.bot-publish {
  width: 100%;
  height: calc(100% - 66px);
  .publish-item {
    width: 512px;
    height: 150px;
    border-radius: 12px;
    background: #fff;
    padding: 28px;
    .content {
      width: 100%;
      height: 56px;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      img {
        width: 56px;
        height: 56px;
        margin-right: 12px;
      }
      .intro {
        .title {
          font-size: 18px;
          font-weight: 500;
          color: #222222;
        }
        .text {
          font-size: 14px;
          color: #999999;
        }
      }
    }
    .bottom {
      // width: 192px;
      margin-left: 68px;
      display: flex;
      // justify-content: space-between;
      .bottom-item {
        font-size: 14px;
        color: #666666;
        margin-right: 20px;
        display: flex;
        align-items: center;
        cursor: pointer;
        img {
          width: 16px;
          height: 16px;
          margin-right: 4px;
        }
      }
    }
  }
}
</style>
