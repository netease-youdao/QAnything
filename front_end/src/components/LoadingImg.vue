<template>
  <img ref="loadingRef" class="img" src="../assets/home/icon-loading.png" alt="loading" />
</template>

<script setup lang="ts">
const loadingRef = ref();
// 解析中img动画，开始动画
const startSpin = () => {
  if (loadingRef.value) {
    loadingRef.value.classList.add('loading');
  }
};

// 停止动画
const stopSpin = () => {
  if (loadingRef.value) {
    loadingRef.value.classList.remove('loading');
  }
};

onMounted(() => {
  startSpin();
});

onUnmounted(() => {
  stopSpin();
});
</script>

<style lang="scss" scoped>
.img {
  width: 100%;
  height: 100%;
}

.loading {
  animation: spin 2s infinite linear;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style>
