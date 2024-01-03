const path = require('path');
const fs = require('fs');
const _inquirer = require('inquirer');
const childProcess = require('child_process');

// 执行cmd命令
function execProcess(cmd, callback) {
  childProcess.exec(cmd, callback);
}
function getPath(filePath) {
  return path.resolve(__dirname, filePath);
}
function getRelativePath(filePath) {
  return path.join(process.cwd(), filePath);
}
// 写文件
function writeFileSync(filePath, fileContent, type = 'utf8') {
  return fs.writeFileSync(filePath, fileContent, type);
}
function mkdirSync(filePath, mode, recursive, context) {
  fs.mkdirSync(filePath, mode, recursive, context);
}
function readFileSync(filePath, type = 'utf-8') {
  return fs.readFileSync(filePath, type); // 读取文件内容
}
function inqucmdirerCmd(prompt, callback) {
  _inquirer.prompt(prompt).then(answers => {
    if (callback) {
      callback(answers);
    }
  });
}
function eslintCheck(filePath) {
  const eslintStr = `${path.join(process.cwd(), 'node_modules/.bin/eslint')} --fix ${filePath}`;
  execProcess(eslintStr, err => {
    if (err) {
      console.log(`eslint修复失败--,失败文件位置：${filePath}`, err);
    } else {
      console.log(`eslint修复成功--`);
    }
  });
}
function readFolder(entryPath, callback) {
  // 递归读取入口文件夹下的所有文件地址
  const files = fs.readdirSync(path.resolve(__dirname, entryPath));
  files.forEach(file => {
    const filePath = path.resolve(__dirname, `${entryPath}/${file}`); // 文件的绝对路径
    const stat = fs.lstatSync(filePath);
    if (stat.isDirectory()) {
      // 是文件夹
      readFolder(filePath, callback);
    } else {
      callback(entryPath, file);
    }
  });
}
module.exports = {
  readFolder,
  eslintCheck,
  getPath,
  writeFileSync,
  mkdirSync,
  readFileSync,
  inqucmdirerCmd,
  execProcess,
  getRelativePath,
};
