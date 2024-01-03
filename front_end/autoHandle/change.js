const path = require('path');
const fs = require('fs');
const pug = require('pug');
const process = require('child_process');

const entry = path.resolve(__dirname, './'); // 处理入口
const recordPath = './pug-stylus-change-record.js'; // 日志记录文件地址

function createFilePathByName(filePath, file, type) {
  const index = file.lastIndexOf('.');
  const fileName = file.substring(0, index);
  return `${filePath}/${fileName}.${type}`;
}
// 执行cmd命令
function execProcess(cmd, callback) {
  process.exec(cmd, (error, stdout, stderr) => {
    if (callback) {
      callback(error, stdout, stderr);
    }
  });
}

// 写文件
function writeFile(filePath, fileContent, type = 'utf8') {
  return fs.writeFileSync(filePath, fileContent, type);
}
function readFile(filePath, type = 'utf-8') {
  return fs.readFileSync(filePath, type); // 读取文件内容
}
// 添加处理记录
function addRecord(filePath, content, isError) {
  // 添加记录
  const fileContent = JSON.parse(readFile(recordPath)); // 读取文件内容
  if (isError) {
    fileContent.error.push({ filePath, content });
  } else {
    fileContent.success.push({ filePath, content });
  }
  writeFile(recordPath, JSON.stringify(fileContent));
}
// 删除文件
function deleteFile(filePath, callback) {
  fs.unlink(filePath, err => {
    addRecord(filePath, '成功：文件删除成功');
    if (callback) {
      callback(err);
    }
  });
}
// 将输入的pug内容转换
function compilePug(pugString, options = { pretty: true }) {
  const templateTagReg = /<\/?template.*?>/g;
  const pugContent = pugString.replace(templateTagReg, '');
  const fn = pug.compile(pugContent, options);
  // 处理编译结果中的#a='#am'类型的字符串为#a
  let html = fn().replace(/#.+?=("|')#.+?("|')/g, str => str.split('=')[0]);
  html = html.replace(/v-else\s*=\s*("|')v-else("|')/g, 'v-else');
  return html;
}
// 将vue文件import进来的styl类型文件修改成引用scss
function handleStylusImport(fileContent) {
  return fileContent.replace(/import\s*("|').+?\.styl("|')/g, str => str.replace('styl', 'scss'));
}
// 将vue或者styl类型的文件进行scss转换
function handleStylus(filePath, outputPath, isDelete) {
  // 转换失败怎么办？
  // stylus example.styl 可将 demo.styl 文件编译成 example.css 文件。
  const cmd = `stylus-conver -i ${filePath} -o ${outputPath}`;
  execProcess(cmd, error => {
    if (!error) {
      // 处理成功且需要删除原始的stylus文件
      addRecord(filePath, `成功：${filePath}转换为${outputPath}成功`);
      if (isDelete) {
        deleteFile(filePath);
      }
      let fileContent = readFile(outputPath);

      // 将/deep/替换为::v-deep(.a)
      fileContent = fileContent.replace(/\/deep\/(.|[\r\n])*?{/g, str => {
        const newStr = str
          .split('/deep/')[1]
          .split('{')[0]
          .replace(/^\s*|\s* $/g, '');
        return `::v-deep(${newStr}){`;
      });
      fileContent = handleStylusImport(fileContent);
      writeFile(outputPath, fileContent);
    } else {
      addRecord(filePath, `失败：${filePath}转换为${outputPath}失败`, true);
    }
  });
}

// 处理vue文件内容
function handleVue(filePath, file) {
  const vuePath = `${filePath}/${file}`;
  let fileContent = readFile(vuePath); // 读取文件内容
  const pugReg = /<template .*lang=("|')pug("|')>(?:.|[\r\n])*<\/template>/g;
  const pugString = fileContent.match(pugReg);
  const stylusReg = /lang=("|')stylus("|')/g;
  // 内容中含有pug代码
  if (pugString) {
    const html = compilePug(pugString[0]);
    fileContent = fileContent.replace(pugReg, `<template>\r${html}\r\n</template>`);
    const status = writeFile(vuePath, fileContent); // 修改vue文件内容
    if (!status) {
      addRecord(vuePath, '成功：vue文件中的pug转换成功');
    }
  }
  if (stylusReg.test(fileContent)) {
    handleStylus(vuePath, vuePath);
  }
  console.log('处理完成--', `${vuePath}文件`);
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
// 获取文件后缀名
function getExtname(allPath) {
  return path.extname(allPath);
}
// 根据文件类型处理文件内容
function handleFile(filePath, file) {
  const allPath = `${filePath}/${file}`;
  const extname = getExtname(allPath);
  if (extname === '.vue') {
    // vue文件处理pug+stylus
    handleVue(filePath, file);
  }
  if (extname === '.styl') {
    // 处理sylus todo转换之后原来引用styl的地方需要手动改成引用新的scss
    handleStylus(allPath, createFilePathByName(filePath, file, 'scss'), true);
  }
  if (extname === '.pug') {
    // 处理pug
    const fileContent = readFile(allPath); // 读取文件内容
    const html = compilePug(fileContent);
    writeFile(createFilePathByName(filePath, file, 'vue'), `<template>\r${html}\r\n</template>`); // 修改vue文件内容
    deleteFile(`${filePath}/${file}`);
  }
}
writeFile(recordPath, JSON.stringify({ success: [], error: [] })); // 创建修改日志文件
readFolder(entry, handleFile); // 从入口开始读取文件

// pug：好处，无需结束标签代码简洁，强制缩进风格统一，支持include方式进行代码复用
// 拥有自己的语法，包括循环、条件控制、定义变量等功能。可以说如果在没有前端框架的年代，这些功能是多么的有诱惑力，但是，近几年React、Vue的出现，已经解决了这些痛点。

// 作用
//   从项目根目录开始，将项目中的vue类型文件里面的pug转换为template，stylus转换为scss
//   将pug类型文件转换为html文件
//   将stylus类型文件转为scss文件

// 使用方法
// 全局安装pug pug-cli :npm install pug pug-cli -g
// 全局安装stylus-conver:npm install stylus-conver -g
// 在项目根目录放置change.js
// 执行node change.js
// 转换完成之后添加scss相关的配置，
// 删除node_modules重新安装，启动项目

// 注意：
//   注意pug类型和styl类型的文件转换时会同名的html和scss类型文件，如果文件所在位置已有同名的html或者scss会被替换
//   转换完成之后可能启动会报错，具体可查看报错信息手动修改或者修改change.js重新生成
