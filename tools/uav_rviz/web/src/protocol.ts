// 二进制遥测帧 / RGB 帧解码（布局见 tools/uav_rviz/protocol.py）。

export interface DepthMeta {
  w: number;
  h: number;
  enc: 'none' | 'u8' | 'u16';
  max_m: number;
  min: number | null;
  center: number | null;
  max: number | null;
  min_range: number | null;
}

export interface FrameMeta {
  seq: number;
  t: number;
  sim_ts: number;
  mode: 'bypass' | 'direct';
  port: number;
  hz: number;
  degraded: boolean;
  fps: number;
  rpc_ms: number;
  rpc_detail: Record<string, number>;
  episode: number;
  reset: boolean;
  pos: [number, number, number];
  agl: number | null;
  quat: [number, number, number, number]; // ENU/FLU (w, x, y, z)
  euler: { roll: number | null; pitch: number | null; yaw: number | null };
  vel: [number, number, number];
  vel_body: [number, number, number];
  speed: number | null;
  camera: { pos: [number, number, number]; quat: [number, number, number, number] };
  collision: { available: boolean; has_collided: boolean; new: boolean; object: string; ts: number };
  shader_warming: boolean;
  no_return: boolean;
  n_points: number;
  depth: DepthMeta;
  goal_err: { horizontal: number | null; vertical: number | null } | null;
  rgb: boolean;
}

export interface TelemetryFrame {
  kind: 'telemetry';
  seq: number;
  t: number;
  flags: number;
  meta: FrameMeta;
  points: Float32Array; // 世界系 ENU XYZ，长度 n*3
  depth: Uint8Array | Uint16Array | null;
  depthW: number;
  depthH: number;
  receivedAt: number;
}

export interface RgbFrame {
  kind: 'rgb';
  seq: number;
  width: number;
  height: number;
  png: Uint8Array;
}

const HEADER = 36;
const decoder = new TextDecoder('utf-8');

export function decodeFrame(buf: ArrayBuffer): TelemetryFrame | RgbFrame | null {
  if (buf.byteLength < 16) return null;
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic === 'UVRT') {
    if (buf.byteLength < HEADER) return null;
    const version = view.getUint16(4, true);
    if (version !== 1) throw new Error(`不支持的遥测协议版本 ${version}`);
    const flags = view.getUint16(6, true);
    const seq = view.getUint32(8, true);
    const t = view.getFloat64(12, true);
    const metaLen = view.getUint32(20, true);
    const nPoints = view.getUint32(24, true);
    const dw = view.getUint16(28, true);
    const dh = view.getUint16(30, true);
    const enc = view.getUint8(32);
    const meta = JSON.parse(decoder.decode(new Uint8Array(buf, HEADER, metaLen))) as FrameMeta;
    let offset = HEADER + metaLen;
    const points = new Float32Array(buf, offset, nPoints * 3);
    offset += nPoints * 12;
    let depth: Uint8Array | Uint16Array | null = null;
    if (enc === 1) depth = new Uint8Array(buf, offset, dw * dh);
    else if (enc === 2) depth = new Uint16Array(buf, offset, dw * dh);
    return { kind: 'telemetry', seq, t, flags, meta, points, depth, depthW: dw, depthH: dh, receivedAt: performance.now() };
  }
  if (magic === 'UVRG') {
    const seq = view.getUint32(4, true);
    const width = view.getUint16(8, true);
    const height = view.getUint16(10, true);
    const len = view.getUint32(12, true);
    return { kind: 'rgb', seq, width, height, png: new Uint8Array(buf, 16, len) };
  }
  return null;
}
