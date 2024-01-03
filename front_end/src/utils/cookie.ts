import Cookies from 'js-cookie';

export function getCookieDomain() {
  let cookieDomain: any = document.domain.split('.');
  if (cookieDomain.length > 1 && isNaN(cookieDomain[cookieDomain.length - 1])) {
    cookieDomain = '.' + cookieDomain.slice(-2).join('.');
  } else {
    cookieDomain = document.domain;
  }
  return cookieDomain;
}

export function setCookie(key, value, other = { expires: 7 }) {
  Cookies.set(key, value, {
    ...other,
  });
}
export function getCookie(key) {
  return Cookies.get(key);
}
export function removeCookie(key, other = { expires: 7 }) {
  Cookies.remove(key, other);
}
