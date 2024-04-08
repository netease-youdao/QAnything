import { useRouter } from 'vue-router';

export const useHeader = defineStore('useHeader', () => {
  const route = useRouter();
  const getRouterPath = () => {
    const path = route.currentRoute.value.path;
    console.log('zj-route', path.split('/'));
    return path.split('/')[1] === 'bots' ? 1 : 0;
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
