import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import type { Connection } from './connection';
import type { FrameMeta } from './protocol';
import { useTopic } from './store';
import type { ColorMode, Layers, LiveStore, Mode } from './store';
import { TelemetryPanel } from './TelemetryPanel';
import { Viewport3D } from './Viewport3D';

const PORTS = [41982, 42072, 42073, 41451];
const LAYER_SWITCHES = [
  { key: 'grid', label: '网格标尺', color: 'grid' },
  { key: 'uav', label: '无人机与三轴 TF', color: 'uav' },
  { key: 'velocity', label: '速度矢量', color: 'velocity' },
  { key: 'cloud', label: '深度点云', color: 'cloud' },
  { key: 'trail', label: '历史轨迹', color: 'trail' },
  { key: 'obstacles', label: '障碍盒', color: 'obstacle' },
  { key: 'roofSeeThrough', label: '顶棚透视', color: 'roof' },
  { key: 'goal', label: '目标圈与指向线', color: 'goal' },
] as const satisfies ReadonlyArray<{ key: keyof Layers; label: string; color: string }>;

function PortSelector({ port, store }: { port: number; store: LiveStore }) {
  const status = useTopic(store, 'status', () => store.status);
  const livePorts = status?.training_ports;
  const evalPorts = status?.evaluation_ports ?? [];
  const ports = livePorts ? [...livePorts, ...evalPorts] : PORTS;
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState('');
  const [invalid, setInvalid] = useState(false);
  const preset = ports.includes(port);

  function select(next: string) {
    if (next === 'custom') {
      setValue(String(port));
      setEditing(true);
      setInvalid(false);
      return;
    }
    setEditing(false);
    if (next === 'auto') {
      store.send({ cmd: 'follow_training_port' });
      return;
    }
    store.send({ cmd: 'set_port', port: Number(next) });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = Number(value);
    if (!/^\d{1,5}$/.test(value) || next < 1 || next > 65535) {
      setInvalid(true);
      return;
    }
    store.send({ cmd: 'set_port', port: next });
    setEditing(false);
  }

  return (
    <div className="port-control">
      <label className="port-label">
        <span>RPC 端口</span>
        <select aria-label="AirSim RPC 端口" value={editing ? 'custom' : status?.auto_port ? 'auto' : String(port)} onChange={(e) => select(e.target.value)}>
          {livePorts && <option value="auto">自动 · {ports.includes(port) ? port : '等待训练'}{evalPorts.includes(port) ? ' · 评估' : ''}</option>}
          {ports.map((p) => <option key={p} value={p}>{p}{livePorts && (evalPorts.includes(p) ? ' · 评估' : ' · 训练')}</option>)}
          {!preset && !status?.auto_port && <option value={port}>{port}</option>}
          <option value="custom">自定义…</option>
        </select>
      </label>
      {editing && (
        <form className="port-popover" onSubmit={submit} noValidate>
          <label htmlFor="custom-port">自定义 AirSim 端口</label>
          <div className="port-input-row">
            <input id="custom-port" autoFocus inputMode="numeric" value={value} aria-invalid={invalid}
              onChange={(event) => { setValue(event.target.value); setInvalid(false); }} />
            <button type="submit">连接</button>
            <button type="button" onClick={() => setEditing(false)}>取消</button>
          </div>
          {invalid && <p role="alert">请输入 1–65535 之间的整数</p>}
        </form>
      )}
    </div>
  );
}

function LayerPanel({ store }: { store: LiveStore }) {
  const layers = useTopic(store, 'layers', () => store.layers);
  const scene = useTopic(store, 'scene', () => store.scene);
  const status = useTopic(store, 'status', () => store.status);

  function setColorMode(value: ColorMode) {
    store.setLayers({ colorMode: value });
    if (store.conn.ws === 'open') store.send({ cmd: 'set_color_mode', value });
  }

  return (
    <aside className="panel layers-panel" aria-label="图层">
      <div className="panel-heading">
        <h2>图层</h2>
        <span className="panel-tag">Z-UP</span>
      </div>
      <div className="layer-list">
        {LAYER_SWITCHES.map(({ key, label, color }) => (
          <div key={key}>
            <label className="layer-row">
              <span className={`layer-icon layer-icon--${color}`} aria-hidden="true" />
              <span>{label}</span>
              <input type="checkbox" checked={layers[key]}
                onChange={(event) => store.setLayers({ [key]: event.target.checked })} />
              <span className="toggle" aria-hidden="true" />
            </label>
            {key === 'cloud' && (
              <div className="cloud-settings">
                <div className="segmented cloud-color" role="group" aria-label="点云着色">
                  <button aria-pressed={layers.colorMode === 'distance'} onClick={() => setColorMode('distance')}>距离 Turbo</button>
                  <button aria-pressed={layers.colorMode === 'height'} onClick={() => setColorMode('height')}>高度 Z</button>
                </div>
                <label className="range-label">
                  <span>点大小</span>
                  <output>{layers.pointSize.toFixed(2)} m</output>
                  <input type="range" min="0.04" max="0.45" step="0.01" value={layers.pointSize}
                    aria-label="点大小" onChange={(event) => store.setLayers({ pointSize: event.target.valueAsNumber })} />
                </label>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="scene-info">
        <button className="button refresh-button" disabled={!status?.connected}
          onClick={() => store.send({ cmd: 'refresh_scene' })}>刷新场景障碍</button>
        <span>{scene?.scan_error ? '场景读取失败' : `${scene?.boxes.length ?? 0} 个障碍对象`}</span>
        {scene?.goal?.source === 'course_episodes' && <span>目标来自上一完成回合</span>}
        {scene?.goal?.source === 'navrl_live' && (
          <span>第 {scene.goal.stage} 档 · 第 {scene.goal.episode} 回合</span>
        )}
        {scene?.goal?.source === 'navrl_evaluation' && (
          <span>第 {scene.goal.stage} 档 · 评估路线 {scene.goal.episode}</span>
        )}
        {scene?.navrl && !scene.goal && status?.connected && <span>{status.evaluation_ports?.includes(status.port) ? '等待评估路线' : '等待训练发布随机终点'}</span>}
      </div>
    </aside>
  );
}

export function App({ store, connection }: { store: LiveStore; connection: Connection }) {
  const status = useTopic(store, 'status', () => store.status);
  const conn = useTopic(store, 'conn', () => store.conn);
  const toasts = useTopic(store, 'toast', () => store.toasts);
  const [meta, setMeta] = useState<FrameMeta | null>(null);
  const [collision, setCollision] = useState(false);
  const [showTelemetry, setShowTelemetry] = useState(true);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const current = store.lastFrame?.meta;
      const fresh = current && store.conn.ws === 'open' && store.status?.connected
        && current.port === store.status.port && current.mode === store.status.mode
        && performance.now() - store.lastFrameAt < 2250;
      setMeta(fresh ? current : null);
      setCollision(Boolean(fresh && (current.collision.has_collided || performance.now() < store.collisionUntil)));
    }, 150);
    connection.start();
    return () => {
      window.clearInterval(timer);
      connection.stop();
    };
  }, [connection, store]);

  const connected = conn.ws === 'open' && Boolean(status?.connected);
  const mode = status?.mode ?? 'bypass';
  const port = status?.port ?? 42072;
  const connectionLabel = connected ? '实时连接'
    : conn.ws !== 'open' ? '连接监控服务'
      : status?.state === 'reconnecting' ? '等待 AirSim 重连' : '正在连接 AirSim';

  function changeMode(next: Mode) {
    if (mode !== next) store.send({ cmd: 'set_mode', mode: next });
  }

  return (
    <div className="app-shell">
      <header className="toolbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /></span>
          <div><h1>UAV RViz</h1><span>AirSim 实时监控</span></div>
        </div>
        <div className="toolbar-source">
          <span className={`connection-state ${connected ? 'is-live' : ''}`} role="status">
            <i aria-hidden="true" />{connectionLabel}
          </span>
          {status?.simulate && <span className="pill simulation-pill">模拟数据</span>}
          <PortSelector store={store} port={port} />
        </div>
        <div className="segmented mode-control" role="group" aria-label="采集模式">
          <button aria-pressed={mode === 'bypass'} onClick={() => changeMode('bypass')}>训练旁路 <small>低开销</small></button>
          <button aria-pressed={mode === 'direct'} onClick={() => changeMode('direct')}>演示直连 <small>20 Hz</small></button>
        </div>
        <div className="toolbar-metrics">
          <span className="metric" title="后端采集帧率"><b>{meta?.fps.toFixed(1) ?? '—'}</b> FPS</span>
          <span className="metric" title="当前帧最慢一次 AirSim RPC"><b>{meta?.rpc_ms.toFixed(0) ?? '—'}</b> ms RPC</span>
          <span className={`pill collision-pill ${collision ? 'is-collision' : ''}`}
            title={collision ? meta?.collision.object || '检测到碰撞' : '从 AirSim 状态读取碰撞信息'} role="status">
            <i aria-hidden="true" />{collision ? '碰撞告警' : meta?.collision.available ? '碰撞监测' : '碰撞未采集'}
          </span>
        </div>
        <div className="toolbar-actions">
          <button className="button" onClick={() => store.fireEvent({ type: 'event', event: 'reset_view' })}>重置视角</button>
          <button className="button" onClick={() => store.send({ cmd: 'clear_trail' })}>清空轨迹</button>
          <button className="button telemetry-toggle" aria-pressed={showTelemetry}
            onClick={() => setShowTelemetry((open) => !open)}>遥测</button>
        </div>
      </header>
      {meta?.shader_warming && (
        <div className="warming-bar" role="status">
          <b>Shader Warming</b><span>深度着色器预热中，点云将在深度有效后显示</span>
        </div>
      )}
      <main className={`workspace ${showTelemetry ? '' : 'workspace--wide'}`}>
        <LayerPanel store={store} />
        <Viewport3D store={store} meta={meta} connected={connected} />
        {showTelemetry && <TelemetryPanel store={store} meta={meta} onClose={() => setShowTelemetry(false)} />}
      </main>
      <footer className="statusbar">
        <span>只读 RPC</span>
        <span className="status-divider" />
        <span>采样 <b>{status?.hz_target ?? 5} Hz</b></span>
        <span>点云网格 <b>{status?.grid.join(' × ') ?? '80 × 45'}</b></span>
        {status?.degraded && <span className="status-warning">RPC 延迟升高，已自动降频</span>}
        {conn.ws === 'open' && status?.error && <span className="status-warning" title={status.error}>AirSim 读取异常</span>}
      </footer>
      <div className="toasts" aria-live="polite">
        {toasts.map((toast) => <div key={toast.id} className={`toast toast--${toast.level}`}>{toast.text}</div>)}
      </div>
    </div>
  );
}
