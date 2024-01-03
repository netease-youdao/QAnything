const { execSync } = require('child_process');
const {
  readFileSync, getRelativePath, writeFileSync,
} = require('./utils');

const branch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' });
function cannotSetVersion(str) {
  if (str.indexOf('.') === -1) {
    return true
  }
  const versionItem = str.split('.');
  const status = versionItem.filter((item) => Number.isNaN(Number(item)));
  if (status.length > 0) {
    return true
  }
  return false
}
// 获取当前git版本号中的数字版本，获取package.json
function getVersion(data = '') {
  let str = data;
  const reg = /.+\/v*/g;
  str = str.replace(reg, '').replace(/[\r\n]/g, '');
  if (cannotSetVersion(str)) return '';

  // 2.2.2
  return str.padEnd(5, '.0');
}
// 根据当前版本号最末位+1
// function getVersionByNow(version) {
//   const versionList = version.split('.');
//   const lastItem = Number(versionList.pop()) + 1;
//   versionList.push(lastItem);
//   return versionList.join('.');
// }
const version = getVersion(branch);

const filePath = getRelativePath('./package.json');
const data = JSON.parse(readFileSync(filePath));
if (!version) return;
// 没有更新
if (data.version !== version) {
  // 从版本号中获取不到类似2.2的变量时，就在当前版本号最末位+1
  data.version = version;
  writeFileSync(filePath, JSON.stringify(data, '', '\t'));
}
// JSON.stringify://https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify
//
