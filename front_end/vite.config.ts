import { defineConfig, loadEnv } from 'vite';
import vue from '@vitejs/plugin-vue';
import eslintPlugin from 'vite-plugin-eslint';
import path from 'path';
import fs from 'fs';
//按需加载antdvue
import Components from 'unplugin-vue-components/vite';
import { AntDesignVueResolver } from 'unplugin-vue-components/resolvers';
//不需要手动引入ref等
import AutoImport from 'unplugin-auto-import/vite';
//自定义svg相关插件
import { createSvgIconsPlugin } from 'vite-plugin-svg-icons';

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
//
const additionalData = (function () {
  let resources = '';
  const styleFolderPath = path.resolve(__dirname, './src/styles/variable');
  readFolder(styleFolderPath, (filePath, file) => {
    const allPath = `@import "@styles/variable/${file}`;
    const extname = getExtname(allPath);
    if (extname === '.scss') {
      resources = `${allPath}";${resources}`; // setting放在前面
    }
  });
  return resources;
})();
const plugins = [] as any;

function resovePath(paths) {
  return path.resolve(__dirname, paths);
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());

  return {
    plugins: [
      Components({
        resolvers: [
          AntDesignVueResolver({
            importStyle: false, // css in js
          }),
        ],
      }),
      AutoImport({
        imports: ['vue', 'vue-router', 'pinia'],
        //下面配置生成自动导入 eslint规则json 生成后enabled改为false，避免重复生成  esint extend导入生成的自动导入json规则
        dts: './auto-imports.d.ts',
        eslintrc: {
          enabled: true,
        },
      }),
      vue(),
      eslintPlugin(),
      createSvgIconsPlugin({
        // 指定需要缓存的图标文件夹
        iconDirs: [path.resolve(process.cwd(), 'src/assets/svg')],
        // 指定symbolId格式
        symbolId: 'icon-[name]',
        // inject: 'body-last' | 'body-first',
        inject: 'body-last',
        customDomId: '__svg__icons__dom__',
      }),

      ...plugins,
    ],
    resolve: {
      // 设置别名
      alias: {
        '@': resovePath('src'),
        '@views/': resovePath('src/views'),
        '@comps': resovePath('./src/components'),
        '@imgs': resovePath('./src/assets/img'),
        '@icons': resovePath('./src/assets/icons'),
        '@utils': resovePath('./src/utils'),
        '@stores': resovePath('./src/store'),
        '@plugins': resovePath('./src/plugins'),
        '@styles': resovePath('./src/styles'),
      },
    },
    css: {
      preprocessorOptions: {
        scss: {
          additionalData,
        },
        less: {
          javascriptEnabled: true,
        },
      },
    },
    build: {
      outDir: `dist/qanything`,
    },

    base: env.VITE_APP_WEB_PREFIX,
    server: {
      usePolling: true,
      port: 5052,
      host: '0.0.0.0',
      open: false,
      fs: {
        strict: false,
      },
      cors: true,
      proxy: {
        [env.VITE_APP_API_PREFIX]: {
		      target: env.VITE_APP_API_HOST + env.VITE_APP_API_PREFIX,
          changeOrigin: true,
		      secure: false,
        },
      },
    },
  };
});
