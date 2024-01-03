// 组件模板
const componentTemplate = `<template>\r
<div class='%fileName%-component'>\r\n</div>
</template>\r\n
<script setup lang='ts'>\r\n
import './index.scss';\r\n
</script>\r\n`;
// 组件模板
const _cssTemplate = '.%fileName%-component {\r\n}\r\n';

const axionsInterceptorTemplate = `
function responseError(err = {}) {//instance
    return err;\r\n
  }\r\n
  function response(res = {}) {//instance
    return res;\r\n
  }\r\n
  function request(config = {}) {//instance
    return config;\r\n
  }\r\n
  return {
    request,
    response,
    responseError,
  }\r\n
`;

module.exports = {
  componentTemplate,
  axionsInterceptorTemplate,
  _cssTemplate,
};
