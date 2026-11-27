// 全局实时数据仓库：状态 / 场景 / 图层配置走 React 订阅；高频帧数据走命令式回调（不触发 React 重渲染）。
import { useSyncExternalStore } from 'react';
import type { FrameMeta, RgbFrame, TelemetryFrame } from './protocol';

export type ColorMode = 'distance' | 'height';
export type Mode = 'bypass' | 'direct';

export interface Layers {
  grid: boolean;
  uav: boolean;
  velocity: boolean;
  cloud: boolean;
  trail: boolean;
  obstacles: boolean;
  roofSeeThrough: boolean;
  goal: boolean;
  pointSize: number;
  colorMode: ColorMode;
  follow: boolean;
}

export const DEFAULT_LAYERS: Layers = {
  grid: true,
  uav: true,
  velocity: true,
  cloud: true,
  trail: true,
  obstacles: true,
  roofSeeThrough: true,
  goal: true,
  pointSize: 0.16,
  colorMode: 'distance',
  follow: true,
};

export interface StatusMsg {
  state: string;
  connected: boolean;
  airsim_host?: string;
  port: number;
  mode: Mode;
  mode_label: string;
  hz_target: number;
  degraded: boolean;
  grid: [number, number];
  rgb: boolean;
  viewers: number;
  error: string | null;
  retry_in: number;
  episode: number;
  shader_warming: boolean;
  rpc_calls: Record<string, number>;
  read_only: boolean;
  simulate?: boolean;
  run_dir?: string | null;
}

export interface SceneBox {
  key: string;
  name: string;
  kind: 'wall' | 'roof' | 'floor' | 'block';
  c: [number, number, number];
  s: [number, number, number];
  q: [number, number, number, number];
  source: string;
}

export interface SceneMsg {
  floor_z: number;
  roof_z: number | null;
  boxes: SceneBox[];
  box_source: string;
  goal: { p: [number, number, number]; radii: [number, number]; alt_tol: number; source: string; family: string | null } | null;
  scan_error: string | null;
}

export interface ConnState {
  ws: 'connecting' | 'open' | 'closed';
  attempts: number;
}

export interface Toast {
  id: number;
  level: 'info' | 'warn' | 'error';
  text: string;
}

export interface History {
  t: number[];
  agl: (number | null)[];
  vx: (number | null)[];
  vy: (number | null)[];
  vz: (number | null)[];
  minRange: (number | null)[];
  gh: (number | null)[];
  gv: (number | null)[];
}

type Topic = 'status' | 'scene' | 'layers' | 'conn' | 'toast';

const LAYERS_KEY = 'uav-rviz.layers.v1';
const HISTORY_CAP = 1500;

function loadLayers(): Layers {
  try {
    const raw = localStorage.getItem(LAYERS_KEY);
    if (raw) return { ...DEFAULT_LAYERS, ...(JSON.parse(raw) as Partial<Layers>) };
  } catch {
    /* localStorage 不可用时使用默认值 */
  }
  return { ...DEFAULT_LAYERS };
}

function emptyHistory(): History {
  return { t: [], agl: [], vx: [], vy: [], vz: [], minRange: [], gh: [], gv: [] };
}

export interface EventMsg {
  type: 'event';
  event: string;
  [k: string]: unknown;
}

export class LiveStore {
  status: StatusMsg | null = null;
  scene: SceneMsg | null = null;
  conn: ConnState = { ws: 'connecting', attempts: 0 };
  layers: Layers = loadLayers();
  toasts: Toast[] = [];
  lastFrame: TelemetryFrame | null = null;
  lastFrameAt = 0;
  collisionUntil = 0;
  clientFps = 0;
  frameCount = 0;
  rgbCount = 0;
  history: History = emptyHistory();
  send: (message: Record<string, unknown>) => void = () => undefined;
  /** 供无头浏览器 / 脚本校验读取 */
  debug = {
    frames: 0,
    pointsDrawn: 0,
    maxPointsDrawn: 0,
    warmingFrames: 0,
    rgbFrames: 0,
    webgl: 'pending' as 'pending' | 'ok' | 'failed',
    errors: [] as string[],
    lastMeta: null as FrameMeta | null,
  };

  private subs = new Map<Topic, Set<() => void>>();
  private frameSubs = new Set<(f: TelemetryFrame) => void>();
  private rgbSubs = new Set<(f: RgbFrame) => void>();
  private eventSubs = new Set<(e: EventMsg) => void>();
  private rxTimes: number[] = [];
  private toastId = 0;

  constructor() {
    (window as unknown as { __uavrviz: unknown }).__uavrviz = this;
    window.addEventListener('error', (e) => this.debug.errors.push(String(e.message)));
    window.addEventListener('unhandledrejection', (e) => this.debug.errors.push(String(e.reason)));
  }

  subscribe(topic: Topic, cb: () => void): () => void {
    let set = this.subs.get(topic);
    if (!set) this.subs.set(topic, (set = new Set()));
    set.add(cb);
    return () => set!.delete(cb);
  }

  private emit(topic: Topic): void {
    this.subs.get(topic)?.forEach((cb) => cb());
  }

  onFrame(cb: (f: TelemetryFrame) => void): () => void {
    this.frameSubs.add(cb);
    return () => this.frameSubs.delete(cb);
  }
  onRgb(cb: (f: RgbFrame) => void): () => void {
    this.rgbSubs.add(cb);
    return () => this.rgbSubs.delete(cb);
  }
  onEvent(cb: (e: EventMsg) => void): () => void {
    this.eventSubs.add(cb);
    return () => this.eventSubs.delete(cb);
  }

  setConn(ws: ConnState['ws'], attempts?: number): void {
    this.conn = { ws, attempts: attempts ?? this.conn.attempts };
    this.emit('conn');
  }

  setStatus(status: StatusMsg): void {
    const changedPort = this.status !== null && this.status.port !== status.port;
    this.status = status;
    if (changedPort) {
      this.lastFrame = null;
      this.lastFrameAt = 0;
      this.collisionUntil = 0;
      this.rxTimes = [];
      this.clientFps = 0;
      this.scene = null;
      this.clearHistory();
      this.emit('scene');
      this.fireEvent({ type: 'event', event: 'source_changed' });
    }
    this.emit('status');
  }

  setScene(scene: SceneMsg): void {
    this.scene = scene;
    this.emit('scene');
  }

  setLayers(patch: Partial<Layers>): void {
    this.layers = { ...this.layers, ...patch };
    try {
      localStorage.setItem(LAYERS_KEY, JSON.stringify(this.layers));
    } catch {
      /* ignore */
    }
    this.emit('layers');
  }

  toast(level: Toast['level'], text: string, ttlMs = 4500): void {
    const id = ++this.toastId;
    this.toasts = [...this.toasts, { id, level, text }].slice(-4);
    this.emit('toast');
    setTimeout(() => {
      this.toasts = this.toasts.filter((t) => t.id !== id);
      this.emit('toast');
    }, ttlMs);
  }

  clearHistory(): void {
    this.history = emptyHistory();
  }

  fireEvent(e: EventMsg): void {
    if (e.event === 'clear_trail') this.clearHistory();
    if (e.event === 'color_mode' && (e.value === 'distance' || e.value === 'height')) {
      if (this.layers.colorMode !== e.value) this.setLayers({ colorMode: e.value });
    }
    this.eventSubs.forEach((cb) => cb(e));
  }

  pushFrame(f: TelemetryFrame): void {
    const m = f.meta;
    if (this.status && (m.port !== this.status.port || m.mode !== this.status.mode)) return;
    this.lastFrame = f;
    this.lastFrameAt = performance.now();
    if (m.collision.new) this.collisionUntil = this.lastFrameAt + 1600;
    this.frameCount++;
    this.debug.frames = this.frameCount;
    this.debug.lastMeta = m;
    if (m.shader_warming) this.debug.warmingFrames++;
    const now = f.receivedAt;
    this.rxTimes.push(now);
    while (this.rxTimes.length > 1 && now - this.rxTimes[0] > 2000) this.rxTimes.shift();
    this.clientFps = this.rxTimes.length > 1 ? ((this.rxTimes.length - 1) * 1000) / (now - this.rxTimes[0]) : 0;

    if (m.reset) this.clearHistory();
    const history = this.history;
    history.t.push(m.t);
    history.agl.push(m.agl);
    history.vx.push(m.vel[0]);
    history.vy.push(m.vel[1]);
    history.vz.push(m.vel[2]);
    history.minRange.push(m.shader_warming ? null : m.depth.min_range);
    history.gh.push(m.goal_err?.horizontal ?? null);
    history.gv.push(m.goal_err?.vertical ?? null);
    const excess = history.t.length - HISTORY_CAP;
    if (excess > 0) {
      for (const key of Object.keys(history) as (keyof History)[]) history[key].splice(0, excess);
    }
    this.frameSubs.forEach((cb) => cb(f));
  }

  pushRgb(f: RgbFrame): void {
    this.rgbCount++;
    this.debug.rgbFrames = this.rgbCount;
    this.rgbSubs.forEach((cb) => cb(f));
  }
}

export function useTopic<T>(store: LiveStore, topic: Topic, get: () => T): T {
  return useSyncExternalStore((cb) => store.subscribe(topic, cb), get);
}
