// Turbo 色阶（Mikhailov 多项式近似），预生成 256 级 LUT。

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function turboRgb(x: number): [number, number, number] {
  const t = clamp01(x);
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [clamp01(r), clamp01(g), clamp01(b)];
}

export const TURBO_LUT: Float32Array = (() => {
  const lut = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = turboRgb(i / 255);
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
})();

/** 写入 out[offset..offset+2]（0..1 浮点），t ∈ [0,1] */
export function turboInto(t: number, out: Float32Array, offset: number): void {
  const i = Math.round(clamp01(t) * 255) * 3;
  out[offset] = TURBO_LUT[i];
  out[offset + 1] = TURBO_LUT[i + 1];
  out[offset + 2] = TURBO_LUT[i + 2];
}

/** 0xAABBGGRR（小端 ImageData 友好）LUT，用于深度伪彩图 */
export const TURBO_U32: Uint32Array = (() => {
  const lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const r = Math.round(TURBO_LUT[i * 3] * 255);
    const g = Math.round(TURBO_LUT[i * 3 + 1] * 255);
    const b = Math.round(TURBO_LUT[i * 3 + 2] * 255);
    lut[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return lut;
})();

export function turboCss(t: number): string {
  const [r, g, b] = turboRgb(t);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}
