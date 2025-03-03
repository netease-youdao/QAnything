import { defineConfig, loadEnv } from 'vite';
import vue from '@vitejs/plugin-vue';
import eslintPlugin from 'vite-plugin-eslint';
import viteImagemin from 'vite-plugin-imagemin';
import { visualizer } from 'rollup-plugin-visualizer';
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
const plugins = [
  // 图片资源压缩
  viteImagemin({
    gifsicle: {
      // gif图片压缩
      optimizationLevel: 3, // 选择1到3之间的优化级别
      interlaced: false, // 隔行扫描gif进行渐进式渲染
    },
    optipng: {
      // png
      optimizationLevel: 7, // 选择0到7之间的优化级别
    },
    mozjpeg: {
      // jpeg
      quality: 20, // 压缩质量，范围从0(最差)到100(最佳)。
    },
    pngquant: {
      // png
      quality: [0.8, 0.9], // Min和max是介于0(最差)到1(最佳)之间的数字，类似于JPEG。达到或超过最高质量所需的最少量的颜色。如果转换导致质量低于最低质量，图像将不会被保存。
      speed: 4, // 压缩速度，1(强力)到11(最快)
    },
  }),
] as any;

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
      visualizer({
        emitFile: false,
        filename: 'analysis-chart.html',
        open: false, // 在打包后是否自动展示
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
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true, // 所有console
          drop_debugger: true, // debugger
        },
      },
    },

    base: env.VITE_APP_WEB_PREFIX,
    server: {
      hmr: true,
      usePolling: true,
      port: 5052,
      host: '0.0.0.0',
      open: false,
      fs: {
        strict: false,
      },
      cors: true,
      proxy: {
        '/local_doc_qa': {
          target: 'https://qanything-dev-gpu214.inner.youdao.com/api',
          changeOrigin: true,
          rewrite: path => path.replace(/^\/local_doc_qa/, '/local_doc_qa'),
        },
        [env.VITE_APP_API_PREFIX]: {
          target: env.VITE_APP_API_HOST,
          changeOrigin: true,
          secure: false,
        },
      },
    },
  };
});
