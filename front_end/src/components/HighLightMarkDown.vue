<template>
  <div class="high-light-comp" v-html="html"></div>
</template>
<script setup lang="ts">
import { nextTick, watch } from 'vue';
import 'highlight.js/styles/stackoverflow-dark.css';
// import showdown from 'showdown';
import hljs from 'highlight.js';
import MarkdownIt from 'markdown-it';

const props = defineProps({
  content: {
    type: String,
    default: '',
  },
  showCode: {
    type: Boolean,
    default: false,
  },
});

// 使用markdown-it实例并开启highlight.js支持
const md = new MarkdownIt({
  html: true,
  breaks: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return (
          '<pre class="hljs"><code>' +
          hljs.highlight(str, { language: lang }).value +
          '</code></pre>'
        );
      } catch (__) {
        console.log('markdown-err');
      }
    }
    return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>'; // 使用默认的转义
  },
});

// const converter = new showdown.Converter();
const html = ref('');
watch(
  () => props.content,
  newvalue => {
    html.value = md.render(newvalue);
  },
  { immediate: true }
);
watch(
  () => props.showCode,
  newvalue => {
    nextTick(() => {
      if (newvalue) {
        document.querySelectorAll('pre code').forEach(() => {
          hljs.highlightAll();
        });
      }
    });
  },
  { immediate: true }
);
</script>
<style>
.high-light-comp {
  user-select: text;
  word-break: break-all;
}

.high-light-comp ul,
.high-light-comp ol {
  padding-left: 20px;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
}

.high-light-comp li {
  list-style-position: inside;
  list-style-type: disc;
  word-break: break-all;
  /* white-space: pre-line; */

  p {
    margin: 0;
    display: inline;
  }
}

.high-light-comp table {
  border-collapse: collapse;
  border-spacing: 0;
}

.high-light-comp tr th {
  background-color: #f0f0f0;
  border: 1px solid #dbdbdb;
  border-bottom: solid 2px #bfbfbf;
  padding: 8px;
  /* text-align: left; */
  margin: 0;
}

.hight-light-comp td {
  border: 1px solid #dbdbdb;
  padding: 4px;
  text-align: center;
}

.hight-light-comp a {
  color: blue;
}

.hljs {
  background: #333;
  color: #f0f0f0;
  padding: 20px;
  overflow-x: auto;
}

.hight-light-comp p {
  word-wrap: break-word;
  line-break: anywhere;
  word-break: break-all;
  white-space: pre-line;
}
</style>
