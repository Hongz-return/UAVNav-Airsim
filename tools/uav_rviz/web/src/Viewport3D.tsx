import { useEffect, useRef } from 'react';
import type { FrameMeta } from './protocol';
import { useTopic } from './store';
import type { LiveStore } from './store';
import { ViewportEngine } from './viewport';

export function Viewport3D({ store, meta, connected }: {
  store: LiveStore;
  meta: FrameMeta | null;
  connected: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const layers = useTopic(store, 'layers', () => store.layers);
  const status = useTopic(store, 'status', () => store.status);

  useEffect(() => {
    const engine = new ViewportEngine(store);
    engine.mount(host.current!);
    return () => engine.dispose();
  }, [store]);

  return (
    <section className="panel viewport-panel" aria-label="3D 场景">
      <div className="viewport-host" ref={host} />
      <div className="viewport-tools">
        <div className="viewport-title">
          <h2>3D 场景</h2>
          <span>{meta ? `${meta.n_points.toLocaleString('zh-CN')} 点` : '等待遥测'}</span>
        </div>
        <div className="segmented view-mode" role="group" aria-label="视角模式">
          <button title="Follow UAV" aria-pressed={layers.follow} onClick={() => store.setLayers({ follow: true })}>跟随无人机</button>
          <button title="Free Orbit" aria-pressed={!layers.follow} onClick={() => store.setLayers({ follow: false })}>自由环绕</button>
        </div>
      </div>
      {!meta && (
        <div className="viewport-wait">
          <b>{connected ? '等待遥测帧' : '等待 AirSim'}</b>
          <span>{status?.state === 'reconnecting' ? '服务会自动重连' : `RPC 端口 ${status?.port ?? 42072}`}</span>
        </div>
      )}
      <div className="viewport-footer">
        <div className="axes-legend" aria-label="坐标轴方向">
          <span className="axis-x">X 前</span><span className="axis-y">Y 左</span><span className="axis-z">Z 上</span>
        </div>
        <span className="orbit-hint">拖动旋转 · 右键平移 · 滚轮缩放</span>
      </div>
    </section>
  );
}
