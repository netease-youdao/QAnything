<template>
  <div class="bots-home-comp">
    <div class="title">
      <span>QAnything </span>
      <span>Bots</span>
    </div>
    <div class="intro">{{ bots.botsHomeDesc1 }}</div>
    <div class="container">
      <p>{{ bots.botsHomeDesc2 }}</p>
      <p>{{ bots.botsHomeDesc3 }}</p>
      <p>{{ bots.botsHomeDesc4 }}</p>
      <div class="new-bots" @click="setNewBotsVisible(true)">
        <img src="@/assets/bots/new-bots.png" alt="new-bots" />
        {{ bots.createBots }}
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import { getLanguage } from '@/language/index';
// import { resultControl } from '@/utils/utils';

const bots = getLanguage().bots;
const { setNewBotsVisible } = useBots();
</script>
<style lang="scss" scoped>
.bots-home-comp {
  width: 80%;
  height: calc(100vh - 64px);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  justify-content: center;

  .title {
    width: 100%;
    text-align: center;
    font-family: Alimama ShuHeiTi;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 12px;

    span:first-child {
      color: #222222;
    }

    span:nth-child(2) {
      color: #5a47e5;
    }
  }

  .intro {
    font-size: 16px;
    text-align: center;
    color: #666666;
    margin-bottom: 40px;
  }

  .container {
    width: 100%;
    height: 400px;
    position: relative;
    background: #fff;
    border-radius: 12px;
    box-sizing: border-box;
    border: 1px dashed #cecece;
    padding: 48px;

    p {
      margin-bottom: 28px;
    }

    .new-bots {
      width: 128px;
      height: 40px;
      border-radius: 6px;
      background: #5a47e5;
      font-size: 16px;
      font-weight: 500;
      color: #fff;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      position: absolute;
      bottom: 42px;
      left: 50%;
      transform: translateX(-50%);

      img {
        width: 20px;
        height: 20px;
        margin-right: 4px;
      }
    }
  }
}
</style>
