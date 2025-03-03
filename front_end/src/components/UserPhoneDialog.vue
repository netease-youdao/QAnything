<template>
  <a-modal
    v-model:open="userPhoneDialogOpen"
    title="请填写手机号"
    :mask-closable="false"
    :keyboard="false"
    destroy-on-close
  >
    <div class="inner">
      <div style="color: red; margin-bottom: 10px">仅做用户标识作用，不会对用户信息做任何收集</div>
      <a-input v-model:value="phone" size="small" style="width: 50%" :status="phoneStatus">
        <template #prefix>
          <PhoneOutlined v-if="phoneStatus !== 'error'" />
          <CloseCircleOutlined v-else />
        </template>
      </a-input>
    </div>
    <template #footer>
      <a-button type="primary" @click="confirmPhone">确认</a-button>
    </template>
  </a-modal>
</template>

<script setup lang="ts">
import { useUser } from '@/store/useUser';
import { CloseCircleOutlined, PhoneOutlined } from '@ant-design/icons-vue';
import message from 'ant-design-vue/es/message';

const { setPhoneNumber } = useUser();
const { userPhoneDialogOpen } = storeToRefs(useUser());

const phoneStatus = ref('');
const phone = ref('');

const checkPhoneRegex = (phone: string): boolean => {
  const regex = /^1[3-9]\d{9}$/;
  return regex.test(phone);
};

const confirmPhone = () => {
  console.log(phone.value);
  if (!checkPhoneRegex(phone.value)) {
    phoneStatus.value = 'error';
    message.error('请输入正确的手机号');
  } else {
    setPhoneNumber(phone.value);
    phoneStatus.value = '';
    message.success('绑定成功');
    userPhoneDialogOpen.value = false;
    window.location.reload();
  }
};

watchEffect(() => {
  if (userPhoneDialogOpen.value) {
    phone.value = '';
  }
});
</script>

<style lang="scss" scoped>
.inner {
  width: 100%;
  display: flex;
  flex-direction: column;
}
</style>
