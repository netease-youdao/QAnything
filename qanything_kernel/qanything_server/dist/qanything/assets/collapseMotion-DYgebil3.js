import{K as c}from"./index-gK4Hs2LI.js";function u(t,s,i,e){for(var n=t.length,l=i+(e?1:-1);e?l--:++l<n;)if(s(t[l],l,t))return l;return-1}function f(t){return t!==t}function h(t,s,i){for(var e=i-1,n=t.length;++e<n;)if(t[e]===s)return e;return-1}function g(t,s,i){return s===s?h(t,s,i):u(t,f,i)}function $(t,s){var i=t==null?0:t.length;return!!i&&g(t,s,0)>-1}function C(t,s,i){for(var e=-1,n=t==null?0:t.length;++e<n;)if(i(s,t[e]))return!0;return!1}const p=t=>({[t.componentCls]:{[`${t.antCls}-motion-collapse-legacy`]:{overflow:"hidden","&-active":{transition:`height ${t.motionDurationMid} ${t.motionEaseInOut},
        opacity ${t.motionDurationMid} ${t.motionEaseInOut} !important`}},[`${t.antCls}-motion-collapse`]:{overflow:"hidden",transition:`height ${t.motionDurationMid} ${t.motionEaseInOut},
        opacity ${t.motionDurationMid} ${t.motionEaseInOut} !important`}}}),x=p,v=Symbol("siderCollapsed"),d=Symbol("siderHookProvider");function r(t,s){return t.classList?t.classList.contains(s):` ${t.className} `.indexOf(` ${s} `)>-1}function o(t,s){t.classList?t.classList.add(s):r(t,s)||(t.className=`${t.className} ${s}`)}function a(t,s){if(t.classList)t.classList.remove(s);else if(r(t,s)){const i=t.className;t.className=` ${i} `.replace(` ${s} `," ")}}const m=function(){let t=arguments.length>0&&arguments[0]!==void 0?arguments[0]:"ant-motion-collapse",s=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0;return{name:t,appear:s,css:!0,onBeforeEnter:i=>{i.style.height="0px",i.style.opacity="0",o(i,t)},onEnter:i=>{c(()=>{i.style.height=`${i.scrollHeight}px`,i.style.opacity="1"})},onAfterEnter:i=>{i&&(a(i,t),i.style.height=null,i.style.opacity=null)},onBeforeLeave:i=>{o(i,t),i.style.height=`${i.offsetHeight}px`,i.style.opacity=null},onLeave:i=>{setTimeout(()=>{i.style.height="0px",i.style.opacity="0"})},onAfterLeave:i=>{i&&(a(i,t),i.style&&(i.style.height=null,i.style.opacity=null))}}},I=m;export{d as S,C as a,u as b,$ as c,I as d,v as e,x as g};
