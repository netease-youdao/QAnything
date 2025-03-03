var getPosition = {
  /**
   * 获取输入光标在页面中的坐标
   * @param        {HTMLElement}   输入框元素
   * @return       {Object}        返回left和top,bottom
   */
  getInputPositon: function (elem) {
    var that = this;
    var cloneDiv = '{$clone_div}',
      cloneLeft = '{$cloneLeft}',
      cloneFocus = '{$cloneFocus}';
    var none = '<span style="white-space:pre-wrap;"> </span>';
    var div = elem[cloneDiv] || document.createElement('div'),
      focus = elem[cloneFocus] || document.createElement('span');
    var text = elem[cloneLeft] || document.createElement('span');
    const res = that._offset(elem);
    console.log(res);
    var index = this._getFocus(elem),
      focusOffset = { left: 0, top: 0 };

    if (!elem[cloneDiv]) {
      (elem[cloneDiv] = div), (elem[cloneFocus] = focus);
      elem[cloneLeft] = text;
      div.appendChild(text);
      div.appendChild(focus);
      document.body.appendChild(div);
      focus.innerHTML = '|';
      focus.style.cssText =
        'display:inline-block;width:0px;overflow:hidden;z-index:-100;word-wrap:break-word;word-break:break-all;';
      div.className = this._cloneStyle(elem);
      div.style.cssText =
        'visibility:hidden;display:inline-block;position:absolute;z-index:-100;word-wrap:break-word;word-break:break-all;overflow:hidden;';
    }
    div.style.left = this._offset(elem).left + 'px';
    div.style.top = this._offset(elem).top + 'px';
    var strTmp = elem.value
      .substring(0, index)
      .replace(/</g, '<')
      .replace(/>/g, '>')
      .replace(/\n/g, '<br/>')
      .replace(/\s/g, none);
    text.innerHTML = strTmp;

    focus.style.display = 'inline-block';
    try {
      focusOffset = this._offset(focus);
      // eslint-disable-next-line no-empty
    } catch (e) {}
    focus.style.display = 'none';
    return {
      left: focusOffset.left,
      top: focusOffset.top,
    };
  },

  // 克隆元素样式并返回类
  _cloneStyle: function (elem, cache) {
    if (!cache && elem['${cloneName}']) return elem['${cloneName}'];
    var className,
      name,
      rstyle = /^(number|string)$/;
    var rname = /^(content|outline|outlineWidth)$/; //Opera: content; IE8:outline && outlineWidth
    var cssText = [],
      sStyle = elem.style;

    let val;
    for (name in sStyle) {
      if (!rname.test(name)) {
        val = this._getStyle(elem, name);
        if (val !== '' && rstyle.test(typeof val)) {
          // Firefox 4
          name = name.replace(/([A-Z])/g, '-$1').toLowerCase();
          cssText.push(name);
          cssText.push(':');
          cssText.push(val);
          cssText.push(';');
        }
      }
    }
    cssText = cssText.join('');
    elem['${cloneName}'] = className = 'clone' + new Date().getTime();
    this._addHeadStyle('.' + className + '{' + cssText + '}');
    return className;
  },

  // 向页头插入样式
  _addHeadStyle: function (content) {
    var style = this._style[document];
    if (!style) {
      style = this._style[document] = document.createElement('style');
      document.getElementsByTagName('head')[0].appendChild(style);
    }
    (style.styleSheet && (style.styleSheet.cssText += content)) ||
      style.appendChild(document.createTextNode(content));
  },
  _style: {},

  // 获取最终样式
  _getStyle:
    'getComputedStyle' in window
      ? function (elem, name) {
          return getComputedStyle(elem, null)[name];
        }
      : function (elem, name) {
          return elem.currentStyle[name];
        },

  // 获取光标在文本框的位置
  _getFocus: function (elem) {
    var index = 0;
    if (elem.selectionStart || elem.selectionStart == '0') {
      // Firefox support
      index = elem.selectionStart;
    }
    return index;
  },

  // 获取元素在页面中位置
  _offset: function (elem) {
    var box = elem.getBoundingClientRect(),
      doc = elem.ownerDocument,
      body = doc.body,
      docElem = doc.documentElement;
    var clientTop = docElem.clientTop || body.clientTop || 0,
      clientLeft = docElem.clientLeft || body.clientLeft || 0;
    var top = box.top + (self.pageYOffset || docElem.scrollTop) - clientTop,
      left = box.left + (self.pageXOffset || docElem.scrollLeft) - clientLeft;
    return {
      left: left,
      top: top,
      right: left + box.width,
      bottom: top + (box.height > 222 ? 222 : box.height),
    };
  },
};

// function getPosition(ctrl) {
//   var p = kingwolfofsky.getInputPositon(ctrl);
//   document.getElementById('show').style.left = p.left + 'px';
//   document.getElementById('show').style.top = p.bottom + 'px';
// }

export default getPosition;
