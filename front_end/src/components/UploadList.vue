<!--
 * @Author: 祝占朋 wb.zhuzp01@rd.netease.com
 * @Date: 2023-11-02 14:40:22
 * @LastEditors: 祝占朋 wb.zhuzp01@rd.netease.com
 * @LastEditTime: 2023-11-08 11:15:14
 * @FilePath: \ai-demo\src\components\UploadList.vue
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<template>
  <div class="">
    <slot name="default"></slot>
  </div>
</template>
<script lang="ts" setup></script>
<style lang="scss" scoped>
.my-list {
  width: 800px;
  height: 460px;
  background: #fbfbfe;
  box-sizing: border-box;
  border: 1px dashed #cecece;
  border-radius: 12px;
}
</style>
