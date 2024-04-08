import { useRouter } from 'vue-router';

export default function routeController() {
  const route = useRouter();
  function changePage(path: string, query = {}) {
    const { refreshCurrent, ...other } = query as any;
    if (refreshCurrent && route.currentRoute.value.path === path) {
      // route.go(0)
      window.location.reload();
    } else {
      route.push({ path, query: other });
    }
  }
  function replace(path, query = {}) {
    console.log('zxx--re--', path, route);
    route.replace({ path, query });
  }
  function openTargetNewPage(link: string) {
    window.open(link, '_blank');
  }
  function openNewPage(path: string, query = {}) {
    const routeData = route.resolve({ path, query });
    window.open(routeData.href, '_blank');
  }
  function getUrlParams(key?: string) {
    const { query = {}, params = {} } = route.currentRoute.value;
    if (key) {
      return query[key] || params[key];
    }
    return {
      ...query,
      ...params,
    };
  }

  function beforeEach(callback: any) {
    // 监听路由切换
    route.beforeEach((...params) => {
      callback(...params);
    });
  }
  function getCurrentRoute() {
    return route.currentRoute;
  }
  function isReady(resolve = () => {}, reject = () => {}) {
    route.isReady().then(resolve, reject);
  }
  return {
    isReady,
    getCurrentRoute,
    getUrlParams,
    changePage,
    beforeEach,
    openNewPage,
    openTargetNewPage,
    replace,
  };
}

export function getOneLevelRoute(route = {}) {
  const { path } = route as any;
  if (path.lastIndexOf('/') !== path.indexOf('/')) {
    // 二级路由时返回一级路由
    return path.substring(0, path.lastIndexOf('/'));
  }
  return path;
}
