import en from './en';
import zh from './zh';
import { useLanguage } from '@/store/useLanguage';

const { language } = storeToRefs(useLanguage());

export function getLanguage() {
  return language.value === 'zh' ? zh : en;
}
