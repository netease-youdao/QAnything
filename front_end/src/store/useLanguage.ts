export const useLanguage = defineStore(
  'useLanguage',
  () => {
    const language = ref('zh');
    const setLanguage = info => {
      language.value = info;
    };
    return {
      language,
      setLanguage,
    };
  },
  {
    persist: {
      storage: localStorage,
    },
  }
);
