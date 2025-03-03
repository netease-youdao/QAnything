import { notification, Button } from 'ant-design-vue';

// import versionFile from '../../version.json';

/**
 * @Author: Ianarua 306781523@qq.com
 * @Date: 2024-09-11 15:21:25
 * @LastEditors: Ianarua 306781523@qq.com
 * @LastEditTime: 2024-09-11 15:21:25
 * @FilePath: front_end/src/utils/version.ts
 * @Description: 读取version，目的用来清除缓存(chatSetting), version读取是package.json里面的version配置
 */

interface IVersionFile {
  version: string;
}

const getVersionFile = async (): Promise<IVersionFile> => {
  try {
    const module = await import('../../version.json');
    return module.default || module;
  } catch (error) {
    console.error('获取不到version', error);
    return { version: '' };
  }
};

// 获取用户缓存的version (旧的)
const getCacheVersion = () => {
  return localStorage.getItem('version') || '';
};

const setCacheVersion = (version: string) => {
  localStorage.setItem('version', version);
};

// 获取dist目录下的version文件(新的)
const getLocalVersion = (versionFile): string => {
  const { version } = versionFile;
  return version || '';
};

// 暴露出的方法
export const checkVersion = async () => {
  const versionFile = await getVersionFile();
  const localVersion = getLocalVersion(versionFile);
  const cacheVersion = getCacheVersion();
  // 判断不相等
  if (localVersion !== cacheVersion) {
    if (cacheVersion) {
      // 缓存有版本（没有则代表是第一次进入）
      openNotification(localVersion, cacheVersion);
    } else {
      setCacheVersion(localVersion);
    }
  }
};

const clearChatSettingStorage = () => {
  localStorage.removeItem('useChatSetting');
};

const openNotification = (localVersion: string, cacheVersion: string) => {
  const key = `key`;
  notification.warning({
    message: `提示`,
    description: `前端版本已从${cacheVersion}更新至${localVersion}。点击下方按钮以同步最新配置，这将清除您之前设置的模型配置缓存。(如果不更新使用可能会出现问题)`,
    btn: () =>
      h(
        Button,
        {
          type: 'primary',
          onClick: () => confirm(key, localVersion),
        },
        { default: () => '更新' }
      ),
    key,
    placement: 'topRight',
  });
};

const confirm = (key: string, localVersion: string) => {
  notification.close(key);
  clearChatSettingStorage();
  setCacheVersion(localVersion);
  window.location.reload();

  notification.success({
    message: `提示`,
    description: '已更新，请重新配置模型',
    duration: 2.5,
    key,
    placement: 'topRight',
  });
};
