let store: any;

if (typeof localStorage === 'undefined' || localStorage === null) {
  console.log('localStorage不可用');
  store = false;
} else {
  const ls = window.localStorage;
  const KEY_PREFIX = 'desk';
  // eslint-disable-next-line no-inner-declarations
  function generateKey(key: string) {
    return `${KEY_PREFIX}-${key}`;
  }
  store = {
    set(_key: string, data: any) {
      const val = JSON.stringify(data);
      const key = generateKey(_key);
      try {
        return ls.setItem(key, val);
      } catch (e: any) {
        if (e.name === 'QuotaExceededError') {
          ls.clear();
          ls.setItem(key, val);
        }
      }
      return null;
    },
    get(key: string) {
      try {
        return JSON.parse(ls.getItem(generateKey(key)) as any);
      } catch (error) {
        console.error(error);
        return null;
      }
    },
    remove(key: string) {
      // eslint-disable-next-line no-param-reassign
      key = generateKey(key);
      try {
        ls.removeItem(key);
        return true;
      } catch (error) {
        return false;
      }
    },
  };
}
const lStorage = store;
export default lStorage;
