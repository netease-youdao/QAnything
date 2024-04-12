<template>
  <Teleport to="body">
    <a-modal
      v-model:open="copyUrlVisible"
      :title="bots.copyLink"
      centered
      :destroyOnClose="true"
      wrap-class-name="select-knowledge-modal"
      :footer="null"
    >
      <div class="copy-url-comp">
        <div class="title">{{ bots.UrlLink }}</div>
        <div class="copy-url">
          <span>{{ webUrl }}</span>
          <a-button class="btn" @click="copyUrl">{{ bots.copy }}</a-button>
        </div>
        <div class="title">{{ bots.qrCode }}</div>
        <div class="qr-code-content">
          <a-qrcode ref="qrcode" :value="webUrl" bgColor="#fff" />
          <a-button class="btn save-btn" @click="saveQrCode">{{ bots.save }}</a-button>
          <a-button class="btn" @click="copyQrcode">{{ bots.copy }}</a-button>
        </div>
      </div>
    </a-modal>
  </Teleport>
</template>
<script lang="ts" setup>
import { useBots } from '@/store/useBots';
import { message } from 'ant-design-vue';
import { getLanguage } from '@/language/index';

const { copyUrlVisible, webUrl } = storeToRefs(useBots());
const qrcode = ref(null);
const bots = getLanguage().bots;

message.config({
  top: `100px`,
  duration: 2,
  maxCount: 3,
});

const copyUrl = () => {
  navigator.clipboard.writeText(webUrl.value).then(() => {
    message.success(bots.copySuccessful);
  });
};

const saveQrCode = async () => {
  const url = await qrcode.value.toDataURL();
  const a = document.createElement('a');
  a.download = 'QRCode.png';
  a.href = url;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

const copyQrcode = async () => {
  try {
    // 将canvas转换为blob
    const url = await qrcode.value.toDataURL();
    const fetchRes = await fetch(url);
    const blob = await fetchRes.blob();
    // 使用ClipboardItem构造函数创建一个新的剪贴板项
    const clipboardItem = new ClipboardItem({ 'image/png': blob });
    // 将剪贴板项写入剪贴板
    await navigator.clipboard.write([clipboardItem]);
    message.success(bots.copySuccessful);
    console.log('Canvas已复制到剪贴板');
  } catch (error) {
    console.error('复制到剪贴板失败:', error);
  }
};
</script>
<style lang="scss" scoped>
.copy-url-comp {
  width: 100%;
  height: 100%;
  font-family: PingFang SC;
  .title {
    font-size: 16px;
    font-weight: 500;
    color: #222222;
    margin-bottom: 13px;
  }
  .btn {
    width: 68px;
    height: 32px;
    box-sizing: border-box;
    border: 1px solid #dfe3eb;
    border-radius: 4px;
    background: #fff;
    color: #5a47e5;
    margin-left: 18px;
  }
  .save-btn {
    border: 1px solid #dfe3eb;
    color: #666666;
  }
  .copy-url {
    width: 100%;
    height: 32px;
    font-size: 14px;
    margin-bottom: 18px;
    color: #666666;
    display: flex;
    align-items: center;
  }
  .qr-code-content {
    width: 100%;
    display: flex;
    align-items: center;
  }
  :deep(.ant-qrcode) {
    canvas {
      width: 134px !important;
      height: 134px !important;
    }
  }
}
</style>
