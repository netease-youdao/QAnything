const { componentTemplate, axionsInterceptorTemplate, _cssTemplate } = require('./template');
const { readFileSync, mkdirSync, writeFileSync, getRelativePath, eslintCheck } = require('./utils');
const inquirer = require('inquirer');
const Rx = require('rxjs');

function log(fileName, filePath, name) {
  console.log(`开始创建${fileName},创建位置：${filePath}，文件名称；${name}`);
}

function getTmplate(temlate, config) {
  const { fileArgs = {}, fileName } = config;
  fileArgs.fileName = fileName;
  let tem = temlate;
  Object.keys(fileArgs).forEach(arg => {
    const reg = new RegExp(`%${arg}%`, 'g');
    // eslint-disable-next-line no-eval
    tem = tem.replace(reg, fileArgs[arg]);
  });
  return tem;
}
// 创建组件
function createComponent(config) {
  const { src, fileName, temlate, cssTemplate } = config;
  console.log('开始创建组件...');
  const foldPath = `${src}/${fileName}`;
  log('组件文件夹', src, fileName);
  mkdirSync(getRelativePath(foldPath));
  log('组件样式文件', `${foldPath}/index.scss`, 'index.scss');
  writeFileSync(getRelativePath(`${foldPath}/index.scss`), getTmplate(cssTemplate, config));
  console.log('开始创建组件文件...');
  log('组件文件', `${foldPath}/index.vue`, 'index.vue');
  writeFileSync(getRelativePath(`${foldPath}/index.vue`), getTmplate(temlate, config));
  console.log('组件创建完成');
}
// 创建页面
function createPage(config) {
  const { src, fileName, temlate, cssTemplate } = config;
  const foldPath = `${src}/${fileName}`;
  log('开始创建页面组件：');
  log('开始创建页面文件夹：', foldPath, fileName);
  mkdirSync(getRelativePath(foldPath));
  log('页面子组件文件夹', foldPath, 'components');
  mkdirSync(getRelativePath(`${foldPath}/components`));
  log('页面图片文件夹', foldPath, 'imgs');
  mkdirSync(getRelativePath(`${foldPath}/imgs`));
  log('页面样式文件', `${foldPath}/index.scss`, 'index.scss');
  writeFileSync(getRelativePath(`${foldPath}/index.scss`), getTmplate(cssTemplate, config));
  log('页面文件', `${foldPath}/index.vue`, 'index.vue');
  writeFileSync(getRelativePath(`${foldPath}/index.vue`), getTmplate(temlate, config));
  setRoutes(config);
}

function setRoutes(config) {
  const { src, fileName, routeType } = config;
  const routersPath = getRelativePath('./src/router/routes.ts');
  let file = readFileSync(routersPath);
  const routesSrc = `@/views/${fileName}/index.vue`;
  const routes = `\r\n{\r\npath:"/${fileName}",\r\nname:"/${fileName}",\r\ncomponent:() =>import("${routesSrc}")\r\n},`;
  if (routeType === 'common') {
    const filePath = `${src}/${fileName}/index.vue`;
    const commonReg = /=\s*\[/g;
    file = file.replace(commonReg, `=[${routes}`);
    writeFileSync(routersPath, file);
    eslintCheck(routersPath);
    eslintCheck(getRelativePath(filePath));
    console.log('页面创建完成');
  } else {
    console.log('页面创建完成');
    console.log('————————————请手动添加子路由——————————');
  }
}
// 创建js
function createJs(config) {
  const { src, fileName } = config;
  console.log('开始创建js文件...');
  const foldPath = `${src}/${fileName}.js`;
  log('js文件', foldPath, fileName);
  writeFileSync(getRelativePath(foldPath), '');
  console.log('js创建完成');
}
// 创建拦截器
function createInterceptor(config) {
  const { src, fileName, temlate } = config;
  const foldPath = `${src}/${fileName}.js`;
  log('拦截器文件', foldPath, fileName);
  writeFileSync(getRelativePath(foldPath), getTmplate(temlate, config));
  console.log('axios拦截器创建成功，开始进行eslint校验');
  eslintCheck(getRelativePath(foldPath));
}

const option = {
  component: {
    name: '组件',
    type: 'component', // 类型
    src: 'src/components', // 创建文件位置
    temlate: componentTemplate, // 默认模板位置
    cssTemplate: _cssTemplate, // 默认css模板位置
    exec: createComponent,
  },
  page: {
    name: '页面',
    type: 'page', // 类型
    src: 'src/views', // 创建文件位置
    temlate: componentTemplate, // 默认模板位置
    cssTemplate: _cssTemplate, // 默认css模板位置
    exec: createPage,
  },
  axiosInterceptor: {
    name: 'axios拦截器',
    type: 'axiosInterceptor', // 类型
    src: 'src/services/axiosInterceptor/interceptors', // 创建文件位置
    temlate: axionsInterceptorTemplate, // 默认模板位置
    exec: createInterceptor,
  },
  js: {
    name: 'js文件',
    type: 'js', // 类型
    src: 'src', // 创建文件位置
    exec: createJs,
  },
};
// 组件类型，名称，位置，

const prompt = [
  {
    type: 'list',
    name: 'type',
    message: '请选择创建文件的类型：:',
    default: 'component',
    choices: Object.keys(option).map(createType => {
      const { name, type } = option[createType];
      return { name, value: type };
    }),
  },
  {
    type: 'input',
    name: 'fileName',
    message: '请输入要创建的文件名称:',
  },
  {
    type: 'input',
    name: 'src',
    message: '请输入要创建的文件位置:',
  },
  {
    type: 'list',
    name: 'routeType',
    message: '请选择页面路由的类型：',
    default: 'common',
    choices: [
      { name: '一级路由', value: 'common' },
      { name: '嵌套路由子级', value: 'child' },
    ],
  },
  // {
  //   type: 'input',
  //   name: 'childRouteName',
  //   message: '请输入父级路由名称：',
  // },
  // {
  //   type: 'confirm',
  //   name: 'test',
  //   message: 'Are you handsome?',
  //   default: true,
  // },
];
let questionName,
  questionIndex = 0;
const answersData = {};
function answer() {
  var prompts = new Rx.Subject();
  inquirer.prompt(prompts).ui.process.subscribe(res => {
    if (typeof res === 'object') {
      answersData[res.name] = res.answer;
      questionName = res.name;
    }
    questionIndex++;
    if (questionName === 'src' && answersData.type !== 'page') {
      getUserInput(answersData);
      prompts.complete();
    } else if (questionName === 'routeType') {
      getUserInput(answersData);
      prompts.complete();
      // if (questionName === '') {

      // } else {
      //   console.log('zxx----res---', res, questionName, answersData);
      //   answer();
      // }
    } else {
      answer();
    }
  });
  prompts.next(prompt[questionIndex]);
}

answer();
// 获取用户命令行输入内容
function getUserInput(answers) {
  // {type:'',fileName:'',src:''}
  const createConfig = option[answers.type];
  const { src, fileName, ...others } = answers;
  if (src && src !== 'undefined') {
    createConfig.src = src;
  }
  createConfig.exec({ ...createConfig, fileName, ...others });
}
// inqucmdirerCmd(prompt, getUserInput);

// 不同类型
// 支持参数传入模板
// utils
