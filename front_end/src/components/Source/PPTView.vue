<template>
  <div class="ppt-view-comp">
    <iframe ref="frame" :src="src" class="iframe-viewer" />
  </div>
</template>

<script setup lang="ts">
// https://git.flyfish.dev/flyfish-group/file-viewer-demo
const props = defineProps<{
  url?: string;
  file?: File;
  name?: string;
}>();

defineProps({
  sourceUrl: {
    type: String,
    required: true,
  },
});

// iframe引用
const frame = ref<HTMLIFrameElement>();
// iframe路径指向构建产物，这里是/，因为放在了public下面
// 如果使用cdn，请使用https://viewer.flyfish.dev
const source = `https://viewer.flyfish.dev/`;
// 查看器的源，当前示例为本源，指定为location.origin即可
const viewerOrigin = location.origin;
// 构建完整url
const src = computed(() => {
  // 文件名称，建议传递，提高体验性
  const name = props.name || '';
  if (props.url) {
    // 直接拼接url
    return `${source}?url=${encodeURIComponent(props.url)}&name=${encodeURIComponent(name)}`;
  } else if (props.file) {
    // 直接拼接来源origin
    return `${source}?from=${encodeURIComponent(viewerOrigin)}&name=${encodeURIComponent(name)}`;
  } else {
    return source;
  }
});

// 发送文件数据
const sendFileData = () => {
  nextTick(() => {
    const viewer = frame.value;
    if (!viewer || !props.file) return;
    viewer.onload = () => viewer.contentWindow?.postMessage(props.file, viewerOrigin);
  });
};

onMounted(() => {
  sendFileData();
});
</script>

<style lang="scss" scoped>
.ppt-view-comp {
  width: 100%;

  .iframe-viewer {
    height: calc(100vh - 2px);
    width: 100%;
    border: 0;
  }
}
</style>
