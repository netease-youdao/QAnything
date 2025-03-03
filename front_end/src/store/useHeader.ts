import { useRouter } from 'vue-router';

export const useHeader = defineStore('useHeader', () => {
  const route = useRouter();
  const pathMap = new Map<string, number>([
    ['statistics', -1],
    ['home', 0],
    ['bot', 1],
    ['quickstart', 2],
  ]);
  const getRouterPath = () => {
    const path = route.currentRoute.value.path;
    console.log('zj-route', path.split('/'));
    return pathMap.get(path.split('/')[1]);
  };

  const navIndex = ref(getRouterPath());
  const setNavIndex = value => {
    navIndex.value = value;
  };

  return {
    navIndex,
    setNavIndex,
  };
});
