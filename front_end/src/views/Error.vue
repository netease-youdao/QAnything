<template>
  <div class="error-page">
    <div class="container">
      <div class="icons">
        <img class="sorry-cry" src="@/assets/login/sorry1.png" alt="icon" />
      </div>
      <div class="content-text">
        <img class="sorry" src="@/assets/login/sorry.png" alt="icon" />
        <div>{{ errorTxt }}</div>
      </div>
    </div>
  </div>
</template>
<script lang="ts" setup>
import { useLogin } from '@/store/useLogin';

const { errorTxt } = storeToRefs(useLogin());
</script>
<style lang="scss" scoped>
.error-page {
  width: 100vw;
  height: 100vh;
  background: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  .container {
    margin-bottom: 60px;
    display: flex;
    align-items: center;
  }
  .icons {
    display: flex;
    align-items: end;
    margin-bottom: 20px;
  }
  .sorry-cry {
    width: 287px;
    height: 256px;
    margin-right: 80px;
  }
  .sorry {
    width: 205.5px;
    height: 64px;
  }
  .content-text {
    font-size: 24px;
  }
}
</style>
