// WebSocket 连接：二进制遥测 / RGB 帧 + JSON 控制，自动指数退避重连。
import { decodeFrame } from './protocol';
import type { EventMsg, LiveStore, SceneMsg, StatusMsg } from './store';

export class Connection {
  private ws: WebSocket | null = null;
  private attempts = 0;
  private timer: number | undefined;
  private closed = false;
  private msgId = 0;

  constructor(private store: LiveStore) {
    store.send = (message) => this.send(message);
  }

  private url(): string {
    return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  }

  start(): void {
    this.closed = false;
    this.open();
  }

  stop(): void {
    this.closed = true;
    window.clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
  }

  private open(): void {
    this.store.setConn('connecting', this.attempts);
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url());
    } catch (err) {
      this.store.debug.errors.push(`WebSocket 创建失败: ${String(err)}`);
      this.scheduleReconnect();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.attempts = 0;
      this.store.setConn('open', 0);
    };
    ws.onmessage = (ev) => {
      if (this.ws === ws) this.onMessage(ev);
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.store.setConn('closed');
      this.scheduleReconnect();
    };
    ws.onerror = () => {
      /* onclose 会随后触发并重连；这里不记 JS 错误 */
    };
  }

  private scheduleReconnect(): void {
    if (this.closed) return;
    this.attempts++;
    const delay = Math.min(5000, 400 * Math.pow(1.6, this.attempts));
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => this.open(), delay);
  }

  send(message: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ id: ++this.msgId, ...message }));
    } else {
      this.store.toast('warn', '尚未连接到监控后端，指令未发送');
    }
  }

  private onMessage(ev: MessageEvent): void {
    const store = this.store;
    try {
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data) as { type: string } & Record<string, unknown>;
        switch (msg.type) {
          case 'hello':
          case 'status':
            store.setStatus(msg as unknown as StatusMsg);
            break;
          case 'scene':
            store.setScene(msg as unknown as SceneMsg);
            break;
          case 'event':
            store.fireEvent(msg as unknown as EventMsg);
            break;
          case 'ack':
            if (msg.ok === false) store.toast('error', `指令 ${String(msg.cmd)} 失败：${String(msg.error)}`);
            break;
          default:
            break;
        }
        return;
      }
      const frame = decodeFrame(ev.data as ArrayBuffer);
      if (frame?.kind === 'telemetry') store.pushFrame(frame);
      else if (frame?.kind === 'rgb') store.pushRgb(frame);
    } catch (err) {
      store.debug.errors.push(`消息处理失败: ${String(err)}`);
      store.toast('error', `消息处理失败：${String(err)}`);
    }
  }
}
