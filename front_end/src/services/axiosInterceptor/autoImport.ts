const path = require('path');
const fs = require('fs');
const childProcess = require('child_process');

const entryName = 'index'; // 生成的拦截器入口文件名称
let content = '';
let exportContent = '\r\nexport default {\r\n';
function getFileName(str: string) {
  const index = str.lastIndexOf('.');
  return str.substring(0, index);
}
// 执行cmd命令
function execProcess(cmd: string, callback: any) {
  childProcess.exec(cmd, callback);
}
// 写文件
function writeFile(filePath: string, fileContent: string, type = 'utf8') {
  return fs.writeFileSync(filePath, fileContent, type);
}
function readFolder(entryPath: string) {
  const files = fs.readdirSync(path.resolve(__dirname, entryPath));
  files.forEach((file: any) => {
    const fileName = getFileName(file);
    if (fileName !== entryName) {
      exportContent = `${exportContent}${fileName},\r\n`;
      content = `${content}\r\n import ${fileName} from './${fileName}';`;
    }
  });
  const file = `${content} ${exportContent}}`;
  const writePath = path.resolve(__dirname, `./interceptors/${entryName}.js`);
  writeFile(writePath, file);
  const cmdStr = `${path.join(process.cwd(), 'node_modules/.bin/eslint')} --fix ${writePath}`;
  function err(err: any) {
    if (err) {
      console.log('eslint修复失败--', err);
    }
    return err;
  }
  execProcess(cmdStr, err);
}
readFolder('./interceptors');
