import { useEffect, useRef } from 'react';
import { TURBO_U32, turboCss } from './colormap';
import type { FrameMeta, TelemetryFrame } from './protocol';
import { useTopic } from './store';
import type { History, LiveStore } from './store';

const HISTORY_SECONDS = 15;
const DEPTH_RAMP = `linear-gradient(to right, ${[1, 0.8, 0.6, 0.4, 0.2, 0].map(turboCss).join(', ')})`;

function metric(value: number | null | undefined, digits = 2): string {
  return value != null && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function DepthCamera({ store, meta }: { store: LiveStore; meta: FrameMeta | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const context = canvas.getContext('2d')!;
    let pixels: ImageData | null = null;
    let rgba: Uint32Array | null = null;

    function paint(frame: TelemetryFrame) {
      const { depth, depthW, depthH } = frame;
      if (frame.meta.shader_warming || !depth || !depthW || !depthH) {
        context.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      if (!pixels || !rgba || pixels.width !== depthW || pixels.height !== depthH) {
        canvas.width = depthW;
        canvas.height = depthH;
        pixels = context.createImageData(depthW, depthH);
        rgba = new Uint32Array(pixels.data.buffer);
      }
      const u16 = depth instanceof Uint16Array;
      for (let i = 0; i < depth.length; i++) {
        const d = depth[i];
        rgba[i] = d === 0 ? 0xff1b1510 : TURBO_U32[255 - (u16 ? d >>> 8 : d)];
      }
      context.putImageData(pixels, 0, 0);
    }

    if (store.lastFrame) paint(store.lastFrame);
    return store.onFrame(paint);
  }, [store]);

  const stats = !meta?.shader_warming ? meta?.depth : null;
  const valid = Boolean(stats?.w && stats.h) && !meta?.no_return;
  return (
    <section className="sensor-section" aria-label="前视深度">
      <div className="section-heading"><h3>前视深度</h3><span className="tech-label">DepthPlanar</span></div>
      <div className="camera-pane">
        <canvas ref={canvasRef} aria-label="DepthPlanar Turbo 伪彩图" className={valid ? '' : 'canvas-hidden'} />
        {valid && (
          <div className="crosshair" aria-hidden="true">
            <i /><span>{metric(stats?.center)} m</span>
          </div>
        )}
        {!valid && <span className="camera-placeholder">{meta?.shader_warming ? '着色器预热中' : meta?.no_return ? '无有效深度回波' : '等待深度帧'}</span>}
      </div>
      <div className="depth-scale"><span>近</span><i style={{ backgroundImage: DEPTH_RAMP }} /><span>{stats?.max_m ?? 25} m</span></div>
      <dl className="depth-stats">
        {([['最近', stats?.min], ['中心', stats?.center], ['最远', stats?.max]] as const).map(([label, value]) => (
          <div key={label}><dt>{label}</dt><dd>{metric(value)} <small>m</small></dd></div>
        ))}
      </dl>
    </section>
  );
}

function RgbImage({ store }: { store: LiveStore }) {
  const imgRef = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const img = imgRef.current!;
    let url: string | null = null;
    const stop = store.onRgb((frame) => {
      const next = URL.createObjectURL(new Blob([new Uint8Array(frame.png)], { type: 'image/png' }));
      img.src = next;
      if (url) URL.revokeObjectURL(url);
      url = next;
    });
    return () => {
      stop();
      if (url) URL.revokeObjectURL(url);
    };
  }, [store]);
  return <div className="camera-pane rgb-camera"><span className="camera-placeholder">等待 RGB 帧</span><img ref={imgRef} alt="实时前视 RGB" /></div>;
}

type SeriesKey = Exclude<keyof History, 't'>;
type Series = { key: SeriesKey; name: string; color: string };
const ALTITUDE: readonly Series[] = [{ key: 'agl', name: 'AGL', color: '#4ddcc6' }];
const VELOCITY: readonly Series[] = [
  { key: 'vx', name: 'X', color: '#fb7984' },
  { key: 'vy', name: 'Y', color: '#5be099' },
  { key: 'vz', name: 'Z', color: '#6ba8ff' },
];
const RANGE: readonly Series[] = [{ key: 'minRange', name: '最近', color: '#f7bd68' }];
const GOAL: readonly Series[] = [
  { key: 'gh', name: '水平', color: '#78b8ff' },
  { key: 'gv', name: '高度', color: '#b795f9' },
];

function historyValue(meta: FrameMeta | null, key: SeriesKey) {
  if (!meta) return null;
  if (key === 'agl') return meta.agl;
  if (key === 'vx') return meta.vel[0];
  if (key === 'vy') return meta.vel[1];
  if (key === 'vz') return meta.vel[2];
  if (key === 'minRange') return meta.shader_warming ? null : meta.depth.min_range;
  return key === 'gh' ? meta.goal_err?.horizontal : meta.goal_err?.vertical;
}

function Trend({ store, title, unit, series, meta, floor, roof }: {
  store: LiveStore;
  title: string;
  unit: string;
  series: readonly Series[];
  meta: FrameMeta | null;
  floor?: number;
  roof?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf: number | null = null;

    function draw() {
      raf = null;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!w || !h) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const data = store.history;
      const end = data.t.at(-1) ?? 0;
      const start = end - HISTORY_SECONDS;
      let first = data.t.length;
      while (first > 0 && data.t[first - 1] >= start) first--;
      let low = Math.min(0, floor ?? 0);
      let high = Math.max(0, roof ?? 0);
      for (const { key } of series) {
        for (let i = first; i < data.t.length; i++) {
          const v = data[key][i];
          if (v == null) continue;
          low = Math.min(low, v);
          high = Math.max(high, v);
        }
      }
      const pad = Math.max(0.25, (high - low) * 0.08);
      low -= pad;
      high += pad;
      const left = 32;
      const right = w - 5;
      const top = 6;
      const bottom = h - 6;
      const x = (t: number) => left + (t - start) / HISTORY_SECONDS * (right - left);
      const y = (v: number) => bottom - (v - low) / (high - low) * (bottom - top);

      ctx.strokeStyle = '#23303c';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 5; i++) {
        const gx = left + (right - left) * i / 5;
        ctx.moveTo(gx, top);
        ctx.lineTo(gx, bottom);
      }
      for (let i = 0; i <= 2; i++) {
        const gy = top + (bottom - top) * i / 2;
        ctx.moveTo(left, gy);
        ctx.lineTo(right, gy);
      }
      ctx.stroke();

      ctx.fillStyle = '#8799ac';
      ctx.font = '9px Consolas, monospace';
      ctx.textBaseline = 'middle';
      ctx.fillText(high.toFixed(1), 3, top + 1);
      ctx.fillText(low.toFixed(1), 3, bottom - 1);

      ctx.setLineDash([3, 3]);
      for (const limit of [floor, roof]) {
        if (limit == null) continue;
        ctx.strokeStyle = limit === roof ? '#a97d49' : '#658391';
        ctx.beginPath();
        ctx.moveTo(left, y(limit));
        ctx.lineTo(right, y(limit));
        ctx.stroke();
      }
      ctx.setLineDash([]);

      ctx.lineWidth = 1.4;
      for (const { key, color } of series) {
        ctx.strokeStyle = color;
        ctx.beginPath();
        let penDown = false;
        for (let i = first; i < data.t.length; i++) {
          const v = data[key][i];
          if (v == null) {
            penDown = false;
          } else if (penDown) {
            ctx.lineTo(x(data.t[i]), y(v));
          } else {
            ctx.moveTo(x(data.t[i]), y(v));
            penDown = true;
          }
        }
        ctx.stroke();
      }
    }

    const schedule = () => {
      if (raf === null) raf = requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(canvas);
    const stopFrame = store.onFrame(schedule);
    const stopEvent = store.onEvent(schedule);
    schedule();
    return () => {
      stopFrame();
      stopEvent();
      observer.disconnect();
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, [store, series, floor, roof]);

  return (
    <section className="trend">
      <div className="trend-heading"><h4>{title}</h4><span>{unit}</span></div>
      <canvas ref={ref} aria-label={`${title}最近 ${HISTORY_SECONDS} 秒波形`} />
      <div className="trend-legend">
        {series.map(({ key, name, color }) => (
          <span key={key}><i style={{ backgroundColor: color }} />{name} <b>{metric(historyValue(meta, key))}</b></span>
        ))}
        {roof != null && <span title="相对地面高度">顶棚 <b>{metric(roof)}</b></span>}
      </div>
    </section>
  );
}

export function TelemetryPanel({ store, meta, onClose }: {
  store: LiveStore;
  meta: FrameMeta | null;
  onClose: () => void;
}) {
  const status = useTopic(store, 'status', () => store.status);
  const scene = useTopic(store, 'scene', () => store.scene);
  const rgbAvailable = status?.mode === 'direct';
  const rgb = rgbAvailable && status?.rgb;
  const roof = scene?.roof_z != null ? scene.roof_z - scene.floor_z : undefined;

  return (
    <aside className="panel telemetry-panel" aria-label="遥测与传感器">
      <div className="panel-heading">
        <h2>遥测与传感器</h2>
        <button className="close-panel" onClick={onClose} aria-label="收起遥测面板" title="收起遥测面板">×</button>
      </div>
      <div className="telemetry-scroll">
        <DepthCamera store={store} meta={meta} />
        <section className="rgb-section">
          <label className="layer-row rgb-switch">
            <span>前视 RGB</span>
            {!rgbAvailable && <small>演示直连可用</small>}
            <input type="checkbox" aria-label="前视 RGB" checked={Boolean(rgb)} disabled={!rgbAvailable}
              onChange={(e) => store.send({ cmd: 'set_rgb', enabled: e.target.checked })} />
            <span className="toggle" aria-hidden="true" />
          </label>
          {rgb && (meta
            ? <RgbImage store={store} key={meta.port} />
            : <div className="camera-pane"><span className="camera-placeholder">等待 RGB 帧</span></div>)}
        </section>
        <section className="state-section" aria-label="飞行状态">
          <div className="section-heading"><h3>飞行状态</h3><span className="tech-label">ENU / Z-UP</span></div>
          <dl className="state-values">
            {(['X', 'Y', 'Z'] as const).map((axis, i) => (
              <div key={axis}><dt className={`axis-${axis.toLowerCase()}`}>{axis}</dt><dd>{metric(meta?.pos[i])} <small>m</small></dd></div>
            ))}
            <div><dt>横滚</dt><dd>{metric(meta?.euler.roll, 1)}<small>°</small></dd></div>
            <div><dt>俯仰</dt><dd>{metric(meta?.euler.pitch, 1)}<small>°</small></dd></div>
            <div><dt>偏航</dt><dd>{metric(meta?.euler.yaw, 1)}<small>°</small></dd></div>
          </dl>
        </section>
        <div className="section-heading chart-section-heading"><h3>实时波形</h3><span className="tech-label">最近 15 秒</span></div>
        <Trend store={store} title="高度 Z" unit="m" series={ALTITUDE} meta={meta} floor={0} roof={roof} />
        <Trend store={store} title="三轴速度" unit="m/s" series={VELOCITY} meta={meta} />
        <Trend store={store} title="最小距障距离" unit="m" series={RANGE} meta={meta} />
        <Trend store={store} title="距目标误差" unit="m" series={GOAL} meta={meta} />
      </div>
    </aside>
  );
}
