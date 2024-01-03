// const { readFileSync, getRelativePath, writeFileSync, eslintCheck } = require('./utils');

// const config = {
//   name: '页面',
//   type: 'page',
//   src: 'src/views',
//   fileName: 't',
//   routeType: 'common',
// };
// const { src, fileName, routeType } = config;

// let file = readFileSync(getRelativePath('src/router/index.ts'));
// const routes = {
//   path: `/${fileName}`,
//   name: fileName,
//   component: () => import(`${src}/${fileName}/index.vue`),
// };
// if (routeType === 'common') {
//   const filePath = `${src}/${fileName}/index.vue`;
//   // const commonReg = new RegExp('routes\\s*=\\s*\\[[\\s\\S]*\\]', 'g');
//   const commonReg = new RegExp('const\\s+routes\\s*=\\s*\\[', 'g');
//   file = file.replace(commonReg, `const routes = [${JSON.stringify(routes)}`);
//   writeFileSync(filePath, file);
//   eslintCheck(getRelativePath(filePath));
// }
