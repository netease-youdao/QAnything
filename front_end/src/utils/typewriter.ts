/*
 * @Author: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @Date: 2024-01-09 15:28:56
 * @LastEditors: 祝占朋 wb.zhuzhanpeng01@mesg.corp.netease.com
 * @LastEditTime: 2024-01-11 10:46:40
 * @FilePath: /QAnything/front_end/src/utils/typewriter.ts
 * @Description:
 */

// 打字机队列
export class Typewriter {
  private queue: string[] = [];
  private consuming = false;
  private timmer: any;
  constructor(private onConsume: (str: string) => void) {}
  // 输出速度动态控制
  dynamicSpeed() {
    const speed = 2000 / this.queue.length;
    if (speed > 200) {
      return 200;
    } else {
      return speed;
    }
  }
  // 添加字符串到队列
  add(str: string) {
    if (!str) return;
    this.queue.push(...str.split(''));
  }
  // 消费
  consume() {
    if (this.queue.length > 0) {
      const str = this.queue.shift();
      str && this.onConsume(str);
    }
  }
  // 消费下一个
  next() {
    this.consume();
    // 根据队列中字符的数量来设置消耗每一帧的速度，用定时器消耗
    this.timmer = setTimeout(() => {
      this.consume();
      if (this.consuming) {
        this.next();
      }
    }, this.dynamicSpeed());
  }
  // 开始消费队列
  start() {
    this.consuming = true;
    this.next();
  }
  // 结束消费队列
  done() {
    this.consuming = false;
    clearTimeout(this.timmer);
    // 把queue中剩下的字符一次性消费
    this.onConsume(this.queue.join(''));
    this.queue = [];
  }
}
