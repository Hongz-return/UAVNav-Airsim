// Three.js 视口引擎（命令式）。世界系 = ENU / Z-up 右手系：X 前 / Y 左 / Z 上。
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { turboInto } from './colormap';
import type { TelemetryFrame } from './protocol';
import type { Layers, LiveStore, SceneMsg } from './store';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const BG = 0x0b0f14;
const MAX_POINTS = 20000;
const MAX_TRAIL = 4000;
const CLOUD_RANGE_M = 20;
const DEFAULT_FLOOR_Z = -2.6;
const DEFAULT_HEIGHT_SPAN = 6.4;

function makeDotTexture(): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.45, 'rgba(255,255,255,0.9)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function makeLabel(): { sprite: THREE.Sprite; set: (text: string) => void } {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 48;
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, fog: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(3.2, 1.2, 1);
  let current = '';
  return {
    sprite,
    set(text: string) {
      if (text === current) return;
      current = text;
      const g = canvas.getContext('2d')!;
      g.clearRect(0, 0, 128, 48);
      g.font = '600 26px "Cascadia Mono", Consolas, monospace';
      g.fillStyle = 'rgba(148,163,184,0.85)';
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, 64, 26);
      texture.needsUpdate = true;
    },
  };
}

const KIND_STYLE: Record<string, { color: number; fill: number; edge: number }> = {
  wall: { color: 0x38bdf8, fill: 0.1, edge: 0.8 },
  roof: { color: 0xf59e0b, fill: 0.34, edge: 0.9 },
  floor: { color: 0x64748b, fill: 0.2, edge: 0.35 },
  block: { color: 0xa78bfa, fill: 0.3, edge: 0.95 },
};

export class ViewportEngine {
  private renderer!: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1200);
  private controls!: OrbitControls;
  private raf = 0;
  private resizeObserver?: ResizeObserver;
  private container?: HTMLElement;
  private unsub: Array<() => void> = [];
  private layers: Layers;
  private sceneMsg: SceneMsg | null = null;

  // 对象
  private gridFine!: THREE.GridHelper;
  private gridCoarse!: THREE.GridHelper;
  private gridGroup = new THREE.Group();
  private labels: Array<{ sprite: THREE.Sprite; set: (t: string) => void; axis: 'x' | 'y'; k: number }> = [];
  private uav = new THREE.Group();
  private uavModel = new THREE.Group();
  private axes!: THREE.AxesHelper;
  private velArrow!: THREE.ArrowHelper;
  private cloud!: THREE.Points;
  private cloudGeom = new THREE.BufferGeometry();
  private cloudPos = new Float32Array(MAX_POINTS * 3);
  private cloudCol = new Float32Array(MAX_POINTS * 3);
  private cloudCount = 0;
  private cloudCam = new THREE.Vector3();
  private cloudMat!: THREE.PointsMaterial;
  private obstacleGroup = new THREE.Group();
  private goalGroup = new THREE.Group();
  private goalPos: THREE.Vector3 | null = null;
  private goalLine: THREE.Line | null = null;
  private goalLineGeom: THREE.BufferGeometry | null = null;
  private trail!: Line2 | null;
  private trailMat!: LineMaterial;
  private trailPts: number[] = [];
  private trailCols: number[] = [];
  private trailDirty = false;
  private trailLastBuild = 0;
  private roofMaterials: THREE.MeshBasicMaterial[] = [];

  // 平滑目标
  private tgtPos = new THREE.Vector3();
  private tgtQuat = new THREE.Quaternion();
  private curPos = new THREE.Vector3();
  private curQuat = new THREE.Quaternion();
  private hasPose = false;
  private lastVel = new THREE.Vector3();
  private floorZ = DEFAULT_FLOOR_Z;

  constructor(private store: LiveStore) {
    this.layers = store.layers;
  }

  mount(container: HTMLElement): void {
    this.container = container;
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
    } catch (err) {
      this.store.debug.webgl = 'failed';
      this.store.debug.errors.push(`WebGL 初始化失败: ${String(err)}`);
      this.store.toast('error', 'WebGL 初始化失败，3D 视口不可用');
      return;
    }
    this.store.debug.webgl = 'ok';
    const r = this.renderer;
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.setClearColor(BG, 1);
    container.appendChild(r.domElement);
    r.domElement.style.display = 'block';
    r.domElement.setAttribute('aria-label', '无人机与点云 WebGL 视口');

    this.scene.background = new THREE.Color(BG);
    this.scene.fog = new THREE.Fog(BG, 70, 240);
    this.buildGrid();
    this.buildUav();
    this.buildCloud();
    this.buildTrail();
    this.scene.add(this.obstacleGroup, this.goalGroup, this.uav);

    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 320;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.zoomSpeed = 0.9;
    this.resetView();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    const s = this.store;
    this.unsub.push(
      s.onFrame((f) => this.onFrame(f)),
      s.subscribe('layers', () => this.setLayers(s.layers)),
      s.subscribe('scene', () => this.setScene(s.scene)),
      s.onEvent((e) => {
        if (e.event === 'clear_trail') this.clearTrail();
        if (e.event === 'reset_view') this.resetView();
        if (e.event === 'source_changed') this.clearSource();
      }),
    );
    this.setLayers(s.layers);
    if (s.scene) this.setScene(s.scene);
    let lastRender = performance.now() - 100;
    const loop = (now: number) => {
      this.raf = requestAnimationFrame(loop);
      const elapsed = now - lastRender;
      const maxFps = s.status?.mode === 'direct' ? 30 : 24;
      if (elapsed < 1000 / maxFps - 1) return;
      lastRender = now;
      this.render(Math.min(0.1, elapsed / 1000));
    };
    this.raf = requestAnimationFrame(loop);
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.unsub.forEach((u) => u());
    this.unsub = [];
    this.resizeObserver?.disconnect();
    this.controls?.dispose();
    this.disposeGroup(this.scene);
    this.trailMat?.dispose();
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.domElement.remove();
    }
  }

  // ---- 构建 ----------------------------------------------------------------
  private buildGrid(): void {
    this.gridFine = new THREE.GridHelper(240, 240, 0x1b2735, 0x141d28);
    this.gridCoarse = new THREE.GridHelper(240, 24, 0x2f4256, 0x253547);
    for (const g of [this.gridFine, this.gridCoarse]) {
      g.rotation.x = Math.PI / 2;
      (g.material as THREE.Material).transparent = true;
      (g.material as THREE.Material).depthWrite = false;
      this.gridGroup.add(g);
    }
    (this.gridFine.material as THREE.Material).opacity = 0.55;
    (this.gridCoarse.material as THREE.Material).opacity = 0.9;
    // 世界轴线
    const axisMat = (c: number) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.55, fog: false });
    const ax = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-400, 0, 0.01), new THREE.Vector3(400, 0, 0.01)]), axisMat(0xef4444));
    const ay = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -400, 0.01), new THREE.Vector3(0, 400, 0.01)]), axisMat(0x22c55e));
    this.gridGroup.add(ax, ay);
    // 10 m 标尺标签
    for (let k = -5; k <= 5; k++) {
      for (const axis of ['x', 'y'] as const) {
        const l = makeLabel();
        l.sprite.renderOrder = 5;
        this.labels.push({ ...l, axis, k });
        this.gridGroup.add(l.sprite);
      }
    }
    this.scene.add(this.gridGroup);
  }

  private buildUav(): void {
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.2, 0.08), new THREE.MeshBasicMaterial({ color: 0x1e293b }));
    const bodyEdge = new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry), new THREE.LineBasicMaterial({ color: 0x94a3b8 }));
    this.uavModel.add(body, bodyEdge);
    const armMat = new THREE.MeshBasicMaterial({ color: 0x334155 });
    for (const sign of [1, -1]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.035, 0.03), armMat);
      arm.rotation.z = (sign * Math.PI) / 4;
      this.uavModel.add(arm);
    }
    const rotorPositions: Array<[number, number, number]> = [
      [0.32, 0.32, 0.03],
      [0.32, -0.32, 0.03],
      [-0.32, 0.32, 0.03],
      [-0.32, -0.32, 0.03],
    ];
    rotorPositions.forEach(([x, y, z]) => {
      const front = x > 0;
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.19, 0.012, 8, 36),
        new THREE.MeshBasicMaterial({ color: front ? 0x5eead4 : 0x64748b }),
      );
      ring.position.set(x, y, z);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.05, 12), new THREE.MeshBasicMaterial({ color: 0xe2e8f0 }));
      hub.rotation.x = Math.PI / 2;
      hub.position.set(x, y, z);
      this.uavModel.add(ring, hub);
    });
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.2, 12), new THREE.MeshBasicMaterial({ color: 0xf87171 }));
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(0.28, 0, 0);
    this.uavModel.add(nose);
    this.uavModel.scale.setScalar(1.8);
    this.axes = new THREE.AxesHelper(1.6); // X 红(前) / Y 绿(左) / Z 蓝(上)
    (this.axes.material as THREE.Material).depthTest = false;
    this.axes.renderOrder = 10;
    this.uav.add(this.uavModel, this.axes);
    this.velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xfacc15, 0.35, 0.2);
    this.scene.add(this.velArrow);
  }

  private buildCloud(): void {
    const posAttr = new THREE.BufferAttribute(this.cloudPos, 3).setUsage(THREE.DynamicDrawUsage);
    const colAttr = new THREE.BufferAttribute(this.cloudCol, 3).setUsage(THREE.DynamicDrawUsage);
    this.cloudGeom.setAttribute('position', posAttr);
    this.cloudGeom.setAttribute('color', colAttr);
    this.cloudGeom.setDrawRange(0, 0);
    this.cloudMat = new THREE.PointsMaterial({
      size: this.layers.pointSize,
      vertexColors: true,
      map: makeDotTexture(),
      alphaTest: 0.04,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
      fog: false,
    });
    this.cloud = new THREE.Points(this.cloudGeom, this.cloudMat);
    this.cloud.frustumCulled = false;
    this.scene.add(this.cloud);
  }

  private buildTrail(): void {
    this.trail = null;
    this.trailMat = new LineMaterial({ color: 0xffffff, linewidth: 3.2, vertexColors: true, transparent: true, opacity: 0.95 });
  }

  // ---- 外部输入 ----------------------------------------------------------------
  setLayers(layers: Layers): void {
    this.layers = layers;
    this.gridGroup.visible = layers.grid;
    this.uav.visible = layers.uav && this.hasPose;
    this.velArrow.visible = layers.velocity && this.lastVel.length() > 0.05;
    this.cloud.visible = layers.cloud;
    this.cloudMat.size = layers.pointSize;
    if (this.trail) this.trail.visible = layers.trail;
    this.obstacleGroup.visible = layers.obstacles;
    this.goalGroup.visible = layers.goal;
    for (const m of this.roofMaterials) m.opacity = layers.roofSeeThrough ? 0.035 : KIND_STYLE.roof.fill;
    this.recolor();
  }

  setScene(scene: SceneMsg | null): void {
    this.sceneMsg = scene;
    this.floorZ = scene?.floor_z ?? DEFAULT_FLOOR_Z;
    this.gridGroup.position.z = this.floorZ + 0.002;
    this.disposeGroup(this.obstacleGroup);
    this.disposeGroup(this.goalGroup);
    this.roofMaterials = [];
    if (!scene) return;
    this.rebuildObstacles(scene);
    this.rebuildGoal(scene);
    this.recolor();
  }

  private disposeGroup(group: THREE.Object3D): void {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        for (const material of Array.isArray(mat) ? mat : mat ? [mat] : []) {
          (material as THREE.MeshBasicMaterial).map?.dispose();
          material.dispose();
        }
      });
    }
  }

  private rebuildObstacles(scene: SceneMsg): void {
    for (const box of scene.boxes) {
      const style = KIND_STYLE[box.kind] ?? KIND_STYLE.block;
      const geom = new THREE.BoxGeometry(box.s[0], box.s[1], box.s[2]);
      const fillMat = new THREE.MeshBasicMaterial({
        color: style.color,
        transparent: true,
        opacity: box.kind === 'roof' && this.layers.roofSeeThrough ? 0.035 : style.fill,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      if (box.kind === 'roof') this.roofMaterials.push(fillMat);
      const mesh = new THREE.Mesh(geom, fillMat);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geom),
        new THREE.LineBasicMaterial({ color: style.color, transparent: true, opacity: style.edge }),
      );
      const holder = new THREE.Group();
      holder.add(mesh, edges);
      holder.position.set(box.c[0], box.c[1], box.c[2]);
      holder.quaternion.set(box.q[1], box.q[2], box.q[3], box.q[0]);
      this.obstacleGroup.add(holder);
    }
  }

  private rebuildGoal(scene: SceneMsg): void {
    const goal = scene.goal;
    this.goalPos = null;
    this.goalLine = null;
    this.goalLineGeom = null;
    if (!goal) return;
    const [gx, gy, gz] = goal.p;
    this.goalPos = new THREE.Vector3(gx, gy, gz);
    const colors = [0x34d399, 0x5eead4];
    goal.radii.forEach((r, i) => {
      const pts: THREE.Vector3[] = [];
      for (let k = 0; k <= 96; k++) {
        const a = (k / 96) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r, 0));
      }
      const ring = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: colors[i % 2], fog: false }),
      );
      ring.position.set(gx, gy, gz);
      const groundRing = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.65, fog: false }),
      );
      groundRing.position.set(gx, gy, this.floorZ + 0.03);
      const cyl = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, goal.alt_tol * 2, 48, 1, true),
        new THREE.MeshBasicMaterial({
          color: colors[i % 2], transparent: true, opacity: i === 0 ? 0.16 : 0.22,
          side: THREE.DoubleSide, depthWrite: false, fog: false,
        }),
      );
      cyl.rotation.x = Math.PI / 2;
      cyl.position.set(gx, gy, gz);
      this.goalGroup.add(ring, groundRing, cyl);
    });
    const pole = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(gx, gy, this.floorZ),
        new THREE.Vector3(gx, gy, Math.max(gz + 5.5, this.floorZ + 12.0)),
      ]),
      new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.9, fog: false }),
    );
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0x6ee7b7, fog: false }),
    );
    core.position.set(gx, gy, gz);

    // 醒目标签：GOAL 终点（无视雾化，高空可见）
    const badge = document.createElement('canvas');
    badge.width = 256;
    badge.height = 72;
    const g = badge.getContext('2d')!;
    g.fillStyle = 'rgba(6, 28, 24, 0.88)';
    g.strokeStyle = '#34d399';
    g.lineWidth = 4;
    g.beginPath();
    g.roundRect(6, 6, 244, 60, 14);
    g.fill();
    g.stroke();
    g.font = '700 30px "Microsoft YaHei", "Segoe UI", sans-serif';
    g.fillStyle = '#6ee7b7';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText('GOAL 终点', 128, 37);
    const badgeTex = new THREE.CanvasTexture(badge);
    const badgeSprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: badgeTex, transparent: true, depthTest: false, depthWrite: false, fog: false }),
    );
    badgeSprite.scale.set(5.2, 1.46, 1);
    badgeSprite.position.set(gx, gy, gz + 2.6);
    badgeSprite.renderOrder = 15;

    // 无人机 -> 终点虚线引导线
    this.goalLineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(gx, gy, gz),
      new THREE.Vector3(gx, gy, gz),
    ]);
    this.goalLine = new THREE.Line(
      this.goalLineGeom,
      new THREE.LineDashedMaterial({
        color: 0x34d399, dashSize: 1.5, gapSize: 0.9, transparent: true, opacity: 0.65, fog: false,
      }),
    );
    this.goalLine.visible = false;
    this.goalGroup.add(pole, core, badgeSprite, this.goalLine);
  }

  clearTrail(): void {
    this.trailPts = [];
    this.trailCols = [];
    this.trailDirty = true;
  }

  private clearSource(): void {
    this.clearTrail();
    this.hasPose = false;
    this.uav.visible = false;
    this.velArrow.visible = false;
    this.cloudCount = 0;
    this.cloudGeom.setDrawRange(0, 0);
    this.store.debug.pointsDrawn = 0;
  }

  resetView(): void {
    const focus = this.hasPose ? this.curPos : new THREE.Vector3(8, 0, 0);
    this.controls.target.copy(focus);
    this.camera.position.set(focus.x - 13, focus.y - 11, focus.z + 8.5);
    this.controls.update();
  }

  private onFrame(f: TelemetryFrame): void {
    const m = f.meta;
    this.tgtPos.set(m.pos[0], m.pos[1], m.pos[2]);
    this.tgtQuat.set(m.quat[1], m.quat[2], m.quat[3], m.quat[0]);
    if (!this.hasPose || m.reset) {
      this.curPos.copy(this.tgtPos);
      this.curQuat.copy(this.tgtQuat);
      if (!this.hasPose) {
        this.hasPose = true;
        this.resetView();
      }
    }
    if (m.reset) this.clearTrail();
    this.lastVel.set(m.vel[0], m.vel[1], m.vel[2]);

    // 点云（着色器预热帧为空点云 → 清空显示，绝不渲染虚假贴脸墙）
    const n = Math.min(f.points.length / 3, MAX_POINTS);
    this.cloudPos.set(f.points.subarray(0, n * 3));
    this.cloudCount = n;
    this.cloudCam.set(m.camera.pos[0], m.camera.pos[1], m.camera.pos[2]);
    this.recolor();
    this.cloudGeom.setDrawRange(0, n);
    (this.cloudGeom.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    this.store.debug.pointsDrawn = n;
    if (n > this.store.debug.maxPointsDrawn) this.store.debug.maxPointsDrawn = n;

    // 轨迹
    if (!m.shader_warming || this.trailPts.length === 0) {
      const last = this.trailPts.length;
      const dx = last ? m.pos[0] - this.trailPts[last - 3] : 1;
      const dy = last ? m.pos[1] - this.trailPts[last - 2] : 1;
      const dz = last ? m.pos[2] - this.trailPts[last - 1] : 1;
      if (dx * dx + dy * dy + dz * dz > 0.0004) {
        this.trailPts.push(m.pos[0], m.pos[1], m.pos[2]);
        const c = new Float32Array(3);
        turboInto((m.speed ?? 0) / 6, c, 0);
        this.trailCols.push(c[0], c[1], c[2]);
        if (this.trailPts.length > MAX_TRAIL * 3) {
          this.trailPts.splice(0, 3);
          this.trailCols.splice(0, 3);
        }
        this.trailDirty = true;
      }
    }
  }

  private recolor(): void {
    const n = this.cloudCount;
    if (!n) return;
    const col = this.cloudCol;
    const pos = this.cloudPos;
    if (this.layers.colorMode === 'distance') {
      const cx = this.cloudCam.x;
      const cy = this.cloudCam.y;
      const cz = this.cloudCam.z;
      for (let i = 0; i < n; i++) {
        const dx = pos[i * 3] - cx;
        const dy = pos[i * 3 + 1] - cy;
        const dz = pos[i * 3 + 2] - cz;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        turboInto(1 - d / CLOUD_RANGE_M, col, i * 3);
      }
    } else {
      const span = this.sceneMsg?.roof_z != null ? Math.max(1, this.sceneMsg.roof_z - this.floorZ) : DEFAULT_HEIGHT_SPAN;
      for (let i = 0; i < n; i++) turboInto((pos[i * 3 + 2] - this.floorZ) / span, col, i * 3);
    }
    (this.cloudGeom.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true;
  }

  private rebuildTrail(): void {
    this.trailDirty = false;
    if (this.trail) {
      this.scene.remove(this.trail);
      this.trail.geometry.dispose();
      this.trail = null;
    }
    if (this.trailPts.length < 6) return;
    const geometry = new LineGeometry();
    geometry.setPositions(this.trailPts);
    geometry.setColors(this.trailCols);
    this.trail = new Line2(geometry, this.trailMat);
    this.trail.computeLineDistances();
    this.trail.frustumCulled = false;
    this.trail.visible = this.layers.trail;
    this.scene.add(this.trail);
  }

  // ---- 渲染循环 ----------------------------------------------------------------
  private render(dt: number): void {
    if (this.hasPose) {
      const k = 1 - Math.exp(-dt * 14);
      this.curPos.lerp(this.tgtPos, k);
      this.curQuat.slerp(this.tgtQuat, k);
      this.uav.position.copy(this.curPos);
      this.uav.quaternion.copy(this.curQuat);
      this.uav.visible = this.layers.uav;

      const speed = this.lastVel.length();
      this.velArrow.visible = this.layers.velocity && speed > 0.05;
      if (this.velArrow.visible) {
        this.velArrow.position.copy(this.curPos);
        this.velArrow.setDirection(this.lastVel.clone().normalize());
        this.velArrow.setLength(Math.min(5, Math.max(0.6, speed * 0.7)), 0.35, 0.2);
      }

      if (this.goalLine && this.goalLineGeom && this.goalPos) {
        const attr = this.goalLineGeom.getAttribute('position') as THREE.BufferAttribute;
        attr.setXYZ(0, this.curPos.x, this.curPos.y, this.curPos.z);
        attr.setXYZ(1, this.goalPos.x, this.goalPos.y, this.goalPos.z);
        attr.needsUpdate = true;
        this.goalLine.computeLineDistances();
        this.goalLine.visible = this.layers.goal;
      }

      if (this.layers.follow) {
        const delta = this.curPos.clone().sub(this.controls.target).multiplyScalar(1 - Math.exp(-dt * 6));
        this.controls.target.add(delta);
        this.camera.position.add(delta);
      }

      // 网格跟随（10 m 吸附）+ 标尺标签
      const sx = Math.round(this.curPos.x / 10) * 10;
      const sy = Math.round(this.curPos.y / 10) * 10;
      this.gridGroup.position.x = sx;
      this.gridGroup.position.y = sy;
      for (const l of this.labels) {
        if (l.axis === 'x') {
          l.sprite.position.set(l.k * 10, -1.8, 0.02);
          // 标签位于网格局部坐标，绝对 X = sx + k*10
          l.set(`X ${sx + l.k * 10}`);
        } else {
          l.sprite.position.set(-1.8, l.k * 10, 0.02);
          l.set(`Y ${sy + l.k * 10}`);
        }
        l.sprite.visible = true;
      }
    }
    if (this.trailDirty && performance.now() - this.trailLastBuild > 90) {
      this.trailLastBuild = performance.now();
      this.rebuildTrail();
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const el = this.container;
    if (!el || !this.renderer) return;
    const w = Math.max(1, el.clientWidth);
    const h = Math.max(1, el.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.trailMat.resolution.set(w, h);
  }
}
